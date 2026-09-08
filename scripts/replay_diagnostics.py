"""Replay logged actions in Habitat-Sim, without policy inference or intervention.

RGB-D and semantic evidence are diagnostic reconstruction, never historical nav_goal.
Run from the server project checkout. All outputs use a fresh untracked run directory.
"""
import argparse, gzip, hashlib, json, os, pathlib, subprocess, sys
from datetime import datetime, timezone
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from av_nav.diagnostic import episode_actions


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--selection',default='configs/diagnostics/hm3dv1_fp12.json')
    ap.add_argument('--run-id',required=True)
    ap.add_argument('--nums',nargs='*',type=int)
    ap.add_argument('--data-root',default='/home/zyq/vlfm/data')
    args=ap.parse_args()
    if pathlib.Path(args.run_id).name!=args.run_id: ap.error('Invalid run ID')
    out=ROOT/'runs'/args.run_id;out.mkdir(parents=True,exist_ok=False)
    import cv2, numpy as np, quaternion, habitat_sim
    data=pathlib.Path(args.data_root)
    selection=json.loads((ROOT/args.selection).read_text())
    cases=[r for r in selection['cases'] if not args.nums or r['num'] in args.nums]
    manifest=dict(started=datetime.now(timezone.utc).isoformat(),argv=sys.argv,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        selection_sha256=sha(ROOT/args.selection),habitat_sim=habitat_sim.__version__,
        source_config='/home/zyq/vlfm/outputs/2026-02-20/08-29-42/.hydra/config.yaml',
        config=dict(width=640,height=480,hfov=79,sensor_height=.88,forward_step=.25,turn_angle=30,allow_sliding=False),
        historical_nav_goal_available=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    scene_hashes={}
    for r in cases:
        print('START',r['num'],r['scene'],r['episode_id'],flush=True)
        caseout=out/f"{r['num']:04d}";caseout.mkdir()
        log=pathlib.Path(r['log']);assert sha(log)==r['log_sha256']
        actions=episode_actions(log.read_text(),r['scene'],r['episode_id'])
        dataset=data/'datasets/objectnav/hm3d/v1/val/content'/f"{r['scene']}.json.gz"
        assert sha(dataset)==r['dataset_sha256']
        ds=json.load(gzip.open(dataset,'rt'));ep=ds['episodes'][int(r['episode_id'])]
        assert pathlib.Path(ep['scene_id']).name.split('.')[0]==r['scene']
        goals=ds['goals_by_category'][r['scene']+'.basis.glb_'+r['category']]
        goal_ids={int(g['object_id']) for g in goals}
        scene=data/'scene_datasets'/ep['scene_id']
        if str(scene) not in scene_hashes:
            scene_hashes[str(scene)]={str(f):sha(f) for f in scene.parent.iterdir() if f.is_file()}
        cfg=habitat_sim.SimulatorConfiguration()
        cfg.scene_id=str(scene);cfg.scene_dataset_config_file=str(data/'scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json')
        cfg.enable_physics=False;cfg.gpu_device_id=0;cfg.allow_sliding=False
        specs=[]
        for name,kind in [('rgb',habitat_sim.SensorType.COLOR),('depth',habitat_sim.SensorType.DEPTH),('semantic',habitat_sim.SensorType.SEMANTIC)]:
            spec=habitat_sim.CameraSensorSpec();spec.uuid=name;spec.sensor_type=kind
            spec.resolution=[480,640];spec.position=[0,.88,0];spec.hfov=79
            specs.append(spec)
        agentcfg=habitat_sim.agent.AgentConfiguration();agentcfg.height=.88;agentcfg.radius=.18
        agentcfg.sensor_specifications=specs
        names={1:'move_forward',2:'turn_left',3:'turn_right',4:'look_up',5:'look_down'}
        agentcfg.action_space={a:habitat_sim.agent.ActionSpec(n,habitat_sim.agent.ActuationSpec(amount=.25 if a==1 else 30)) for a,n in names.items()}
        sim=habitat_sim.Simulator(habitat_sim.Configuration(cfg,[agentcfg]));sim.seed(100)
        state=habitat_sim.AgentState();state.position=ep['start_position']
        q=ep['start_rotation'];state.rotation=quaternion.quaternion(q[3],q[0],q[1],q[2])
        agent=sim.initialize_agent(0,state)
        objects={}
        for obj in sim.semantic_scene.objects:
            if obj is None:continue
            i=int(obj.id.split('_')[-1]);objects[i]=dict(id=i,name=obj.category.name(),center=np.asarray(obj.aabb.center).tolist(),sizes=np.asarray(obj.aabb.sizes).tolist(),task_goal=i in goal_ids)
        assert all(i in objects for i in goal_ids)
        cap=cv2.VideoCapture(r['video']);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames=set(int(x) for x in np.linspace(0,n-1,9))
        frames.update([max(0,n-21),max(0,n-6),n-1])
        if r['first_navigate_step'] is not None:
            frames.update([max(0,min(n-1,r['first_navigate_step']+d)) for d in [-1,0,1]])
        # Save neighbouring states to establish video alignment rather than assuming it.
        capture_steps={f+o for f in frames for o in [0,1,2] if 0<=f+o<=len(actions)}
        poses=[];captured={};comparisons=[]
        obs=sim.get_sensor_observations()
        for step in range(len(actions)+1):
            if step>0:
                a=actions[step-1][2]
                if a!=0:obs=sim.step(a)
            st=agent.get_state();sensor=st.sensor_states['depth']
            pose=dict(step=step,position=st.position.tolist(),rotation=quaternion.as_float_array(st.rotation).tolist(),sensor_position=sensor.position.tolist(),sensor_rotation=quaternion.as_float_array(sensor.rotation).tolist())
            poses.append(pose)
            if step in capture_steps:
                rgb=cv2.cvtColor(obs['rgb'][:,:,:3],cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(caseout/f'replay_{step:04d}.jpg'),rgb,[cv2.IMWRITE_JPEG_QUALITY,95])
                np.savez_compressed(caseout/f'sensors_{step:04d}.npz',depth=obs['depth'],semantic=obs['semantic'])
                captured[step]=cv2.resize(rgb,(592,444))
        for f in sorted(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read();assert ok
            cv2.imwrite(str(caseout/f'original_{f:04d}.jpg'),img)
            h=img.shape[0];ph=int((h-72)/2);pw=int(ph*4/3)
            crop=cv2.resize(img[h-ph:,:pw],(592,444))
            gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).astype(float)
            scores=[]
            for offset in [0,1,2]:
                if f+offset not in captured:continue
                gr=cv2.cvtColor(captured[f+offset],cv2.COLOR_BGR2GRAY).astype(float)
                scores.append(dict(offset=offset,mae=float(np.mean(np.abs(gray-gr))),correlation=float(np.corrcoef(gray.ravel(),gr.ravel())[0,1])))
            comparisons.append(dict(video_frame=f,scores=scores))
        cap.release()
        final=poses[-1]['position']
        distance=habitat_sim.MultiGoalShortestPath();distance.requested_start=np.asarray(final)
        distance.requested_ends=np.asarray([v['agent_state']['position'] for g in goals for v in g['view_points']])
        found=sim.pathfinder.find_path(distance)
        result=dict(selection=r,episode=ep,task_goals=goals,objects=objects,actions=actions,poses=poses,comparisons=comparisons,
            video_frames=n,historical_distance=r['distance_to_goal'],replayed_distance=float(distance.geodesic_distance) if found else None,
            source_hashes=scene_hashes[str(scene)],video_sha256=sha(r['video']),historical_nav_goal_available=False,
            note='Action replay only; sampled RGB-D points are diagnostic proxies, not recovered historical policy nav_goal.')
        (caseout/'evidence.json').write_text(json.dumps(result,indent=2))
        sim.close();print('DONE',r['num'],'distance',result['replayed_distance'],'old',r['distance_to_goal'],flush=True)
    (out/'exit.json').write_text(json.dumps(dict(completed=len(cases),returncode=0)))


if __name__=='__main__':main()
