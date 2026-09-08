"""Offline image alignment and semantic pixel audit of a fixed action replay."""
import argparse,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from av_nav.diagnostic import camera_point


def main():
    import cv2,numpy as np,quaternion
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);ap.add_argument('--analysis-id',default='analysis_v2');args=ap.parse_args()
    root=ROOT/'runs'/args.run_id;out=root/args.analysis_id;out.mkdir(exist_ok=False)
    points=json.loads((ROOT/'configs/diagnostics/hm3dv1_fp12_pixels.json').read_text())['cases']
    summaries=[]
    for spec in points:
        case=root/f"{spec['num']:04d}";e=json.loads((case/'evidence.json').read_text());step=spec['replay_step']
        source=cv2.imread(str(case/f"original_{spec['video_frame']:04d}.jpg"))
        replay=cv2.imread(str(case/f'replay_{step:04d}.jpg'))
        # Composite-video RGB panel sizes vary by frame. Calibrate its crop against
        # the replay image, keeping the original video itself intact as evidence.
        gray=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY)
        ref=cv2.resize(cv2.cvtColor(replay,cv2.COLOR_BGR2GRAY),(160,120)).astype(float)
        ref-=ref.mean();rn=np.linalg.norm(ref)
        best=(-2,None)
        for width in range(240,min(800,source.shape[1])+1,4):
            height=round(width*.75)
            for bottom in range(0,16,2):
                top=source.shape[0]-bottom-height
                if top<0:continue
                patch=cv2.resize(gray[top:top+height,:width],(160,120)).astype(float);patch-=patch.mean()
                corr=float(np.sum(patch*ref)/(np.linalg.norm(patch)*rn+1e-10))
                if corr>best[0]:best=(corr,(0,top,width,height,bottom))
        _,top,width,height,bottom=best[1]
        crop=cv2.resize(source[top:top+height,:width],(640,480))
        # Verify alternative one-step offsets using the SAME crop.
        offsets=[]
        for offset in [0,1,2]:
            p=case/f"replay_{spec['video_frame']+offset:04d}.jpg"
            if p.exists():
                im=cv2.imread(str(p));g=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(160,120))
                a=cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(160,120))
                offsets.append(dict(offset=offset,correlation=float(np.corrcoef(a.ravel(),g.ravel())[0,1])))
        sensors=np.load(case/f'sensors_{step:04d}.npz');depth=sensors['depth'];semantic=sensors['semantic']
        pose=e['poses'][step];rotation=quaternion.as_rotation_matrix(np.quaternion(*pose['sensor_rotation']))
        position=np.asarray(pose['sensor_position']);objects=e['objects'];probes=[]
        overlay=replay.copy()
        for j,(u,v) in enumerate(spec['pixels']):
            ids=semantic[max(0,v-2):v+3,max(0,u-2):u+3];vals,counts=np.unique(ids,return_counts=True)
            sid=int(semantic[v,u]);obj=objects.get(str(sid))
            dep=float(depth[v,u]);world=None
            if np.isfinite(dep) and dep>0:world=(rotation@camera_point(u,v,dep)+position).tolist()
            if obj is not None:
                center=np.asarray(obj['center']);half=np.asarray(obj['sizes'])/2
                residual=np.maximum(np.abs(np.asarray(world)-center)-half,0) if world is not None else None
                bbox_distance=float(np.linalg.norm(residual)) if residual is not None else None
                included_floor=bool(e['episode']['start_position'][1]<=center[1]<e['episode']['start_position'][1]+2)
            else:bbox_distance=None;included_floor=None
            probes.append(dict(pixel=[u,v],semantic_id=sid,semantic_name=obj['name'] if obj else None,
                task_goal=obj['task_goal'] if obj else None,depth_m=dep,within_vlfm_depth_range=.5<=dep<=5,
                world_surface_point=world,distance_to_semantic_aabb=bbox_distance,
                object_center_in_initial_floor_mask=included_floor,
                patch_semantic_counts={str(int(k)):int(c) for k,c in zip(vals,counts)},
                point_kind='diagnostic visible-surface point, NOT historical nav_goal'))
            color=(0,200,0) if obj and obj['task_goal'] else (0,120,255)
            cv2.drawMarker(overlay,(u,v),color,cv2.MARKER_CROSS,18,2)
            label=f"P{j+1} ID={sid} {obj['name'] if obj else 'unknown'} goal={obj['task_goal'] if obj else '?'}"
            cv2.putText(overlay,label,(8,25+j*27),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
        semantic_overlay=replay.copy()
        goal_ids=[int(i) for i,o in objects.items() if o['task_goal']]
        target_mask=np.isin(semantic,goal_ids)
        semantic_overlay[target_mask]=(.4*semantic_overlay[target_mask]+.6*np.array([0,255,0])).astype(np.uint8)
        for probe in probes:
            m=semantic==probe['semantic_id']
            if probe['semantic_id']>0 and not probe['task_goal']:
                semantic_overlay[m]=(.4*semantic_overlay[m]+.6*np.array([0,120,255])).astype(np.uint8)
        sheet=np.full((1020,1280,3),255,np.uint8)
        sheet[30:510,:640]=crop;sheet[30:510,640:]=replay
        sheet[540:1020,:640]=overlay;sheet[540:1020,640:]=semantic_overlay
        for txt,pos in [('Original video RGB (crop)',(8,22)),('Action replay RGB',(648,22)),('Manual probe: visible surface',(8,532)),('Green: task goals; orange: probed non-goal',(648,532))]:
            cv2.putText(sheet,txt,pos,cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
        cv2.imwrite(str(out/f"{spec['num']:04d}.jpg"),sheet,[cv2.IMWRITE_JPEG_QUALITY,92])
        summary=dict(num=spec['num'],scene=e['selection']['scene'],episode_id=e['selection']['episode_id'],category=e['selection']['category'],
            selection_group=e['selection']['selection_group'],selection_reason=e['selection']['selection_reason'],
            video_frame=spec['video_frame'],replay_step=step,alignment_correlation=best[0],crop=best[1],offset_comparisons=offsets,
            historical_distance=e['historical_distance'],replayed_distance=e['replayed_distance'],distance_delta=e['replayed_distance']-e['historical_distance'],
            steps=len(e['actions']),stop_called=e['selection']['stop_called'],probes=probes,
            historical_nav_goal_available=False,task_goal_ids=goal_ids)
        summaries.append(summary);print(json.dumps(summary),flush=True)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
