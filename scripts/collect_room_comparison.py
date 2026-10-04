"""Render qualitative paired room maps; never invent room-footprint accuracy."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay-dir', required=True)
    p.add_argument('--active-dir', required=True)
    p.add_argument('--occusg-dir', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np
    replay, active, occusg, output = map(Path, [args.replay_dir,args.active_dir,args.occusg_dir,args.output_dir])
    output.mkdir(parents=True,exist_ok=False)
    manifest = json.loads((replay / 'manifest.json').read_text())
    fig, axes = plt.subplots(len(manifest['cases']), 2, figsize=(12,4*len(manifest['cases'])), squeeze=False)
    rows, details = [], []
    color_map = ListedColormap(plt.cm.tab20(np.linspace(0,1,20)))
    for index, case in enumerate(manifest['cases']):
        # Replay may contain selected smoke cases; directory identity is explicit.
        scene = Path(case['episode']['scene_id']).name.split('.')[0]
        row = dict(scene=scene, source_episode_id=case['selection']['episode_id'],
                   source_index=case['selection']['source_index'], frames=case['frames'])
        start = np.asarray(case['episode']['start_position'])
        paths = np.asarray([pose['position'] for pose in case['poses']])
        maps = []
        for name, folder in [('Active',active/f'{index:02d}'),('OccuSG',occusg/f'{index:02d}')]:
            summary_file = folder/'summary.json'
            if not summary_file.exists():
                row[name+'_status'] = 'failed_or_incomplete'
                maps.append(None)
                continue
            summary = json.loads(summary_file.read_text())
            if json.loads((folder/'exit.json').read_text())['returncode'] != 0:
                raise ValueError(f'Nonzero method exit in {folder}')
            if summary['source']['selection'] != case['selection'] or summary['frames'] != case['frames']:
                raise ValueError(f'Paired episode/frame mismatch in {folder}')
            checkpoint_file = folder/f'checkpoint_{case["frames"]:04d}.npz'
            checkpoint = np.load(checkpoint_file)
            labels = checkpoint['labels']
            if name == 'Active':
                occupied, explored = checkpoint['occupied'], checkpoint['explored']
                known, free = explored > 0, (explored > 0) & (occupied == 0)
                # Native cells are (y=-world_x, x=-world_z), 5cm center samples.
                known, free, labels = [a[::-1,:].T for a in [known,free,labels]]
                h,w = checkpoint['labels'].shape
                extent = [24+start[0]-(h-.5)*.05,24+start[0]+.025,
                          -24-start[2]-.025,-24-start[2]+(w-.5)*.05]
                resolution = .05
                row['Active_door_trigger_frames'] = summary['door_trigger_frames']
                row['Active_final_detected_doors'] = summary['final_detected_doors']
            else:
                occupancy = checkpoint['occupancy']
                known, free = occupancy >= 0, occupancy == 0
                resolution = float(checkpoint['resolution'])
                ox,oy = checkpoint['origin']
                extent = [ox,ox+known.shape[1]*resolution,oy,oy+known.shape[0]*resolution]
            row[name+'_status'] = 'complete'
            row[name+'_regions'] = summary['final_room_count']
            row[name+'_known_area_m2'] = float(known.sum()*resolution**2)
            row[name+'_labeled_free_area_m2'] = float(((labels>0)&free).sum()*resolution**2)
            row[name+'_pipeline_seconds'] = sum(r['seconds'] for r in summary['records'])
            row[name+'_accuracy'] = None
            details.append(dict(scene=scene,method=name,summary=summary))
            maps.append(dict(known=known,free=free,labels=labels,extent=extent))
        bounds = [paths[:,0].min(),paths[:,0].max(),(-paths[:,2]).min(),(-paths[:,2]).max()]
        for data in maps:
            if data is None:
                continue
            coords = np.argwhere(data['known'])
            if coords.size:
                e = data['extent']; h,w = data['known'].shape
                x = e[0] + (coords[:,1]+.5)*(e[1]-e[0])/w
                y = e[2] + (coords[:,0]+.5)*(e[3]-e[2])/h
                bounds = [min(bounds[0],x.min()),max(bounds[1],x.max()),min(bounds[2],y.min()),max(bounds[3],y.max())]
        for column,(name,data) in enumerate(zip(['Active','OccuSG'],maps)):
            ax = axes[index,column]
            if data is None:
                ax.text(.5,.5,'FAILED / INCOMPLETE',ha='center',transform=ax.transAxes)
            else:
                background = np.full(data['known'].shape,.45)
                background[data['known']] = 0
                background[data['free']] = 1
                ax.imshow(background,origin='lower',extent=data['extent'],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                colors = np.ma.masked_where((data['labels']==0)|~data['free'],(data['labels']-1)%20)
                ax.imshow(colors,origin='lower',extent=data['extent'],cmap=color_map,vmin=0,vmax=19,alpha=.8,interpolation='nearest')
            ax.plot(paths[:,0],-paths[:,2],color='red',lw=1,label='shared trajectory')
            ax.scatter([start[0]],[-start[2]],c='red',marker='*',s=70)
            ax.set_xlim(bounds[0]-.5,bounds[1]+.5); ax.set_ylim(bounds[2]-.5,bounds[3]+.5)
            ax.set_aspect('equal'); ax.set_title(f'{scene} | {name} | {row.get(name+"_regions","?")} regions | {case["frames"]} frames')
            ax.set_xlabel('world x (m)'); ax.set_ylabel('-world z (m)')
        rows.append(row)
    fig.suptitle('Same RGB-D / poses: qualitative room segmentation\nColors identify method-local regions; no verified room footprint GT\nGray: unknown | Black: occupied | White: unassigned free | Red: trajectory',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.97])
    fig.savefig(output/'comparison.png',dpi=140)
    plt.close(fig)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (output/'episodes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)
    summary = dict(episodes=rows, paired_complete=sum(r.get('Active_status')==r.get('OccuSG_status')=='complete' for r in rows),
                   accuracy_available=False, notes=['No verified room-footprint GT; region count is not accuracy.',
                   'Known/labeled areas use each native map and are descriptive, not a shared-GT quality score.',
                   'OccuSG pipeline time includes ROS transfer/wait; Active time is serial local computation.',
                   'Shared ObjectNav trajectories; Active autonomous exploration is disabled.'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    (output/'details.json').write_text(json.dumps(details,indent=2))
    (output/'report.md').write_text('# Room segmentation comparison\n\n'+json.dumps(summary,indent=2)+'\n\n![Paired maps](comparison.png)\n')


if __name__=='__main__':
    main()
