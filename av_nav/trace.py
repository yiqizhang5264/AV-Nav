"""Read-only diagnostic instrumentation shared by baseline and AV adapters."""
import gzip,json,os,time
from pathlib import Path
import numpy as np
from .geometry import observe


def native(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if hasattr(x,'tolist'):return x.tolist()
    raise TypeError(type(x).__name__)


class TraceSAM:
    def __init__(self,model,owner):self.model,self.owner=model,owner
    def segment_bbox(self,rgb,bbox):
        mask=self.model.segment_bbox(rgb,bbox)
        owner=self.owner;rgbd=owner._trace_rgbd
        obs=observe(rgb,rgbd[1],mask,*rgbd[2:],owner._num_steps)
        owner._trace_segments.append(dict(bbox=np.asarray(bbox).tolist(),
            center=obs.center.tolist() if obs else None,quality=obs.quality if obs else None,
            mask_pixels=int(np.count_nonzero(mask))))
        return mask


class TraceMixin:
    def __init__(self,*args,**kwargs):
        self._trace_episode=-1;self._trace_sources=[]
        if os.environ.get('AV_DATASET_FILE'):
            with gzip.open(os.environ['AV_DATASET_FILE'],'rt') as f:self._trace_sources=json.load(f)['episodes']
        self._trace_episode_reset()
        super().__init__(*args,**kwargs)
        self._mobile_sam=TraceSAM(self._mobile_sam,self)

    def _trace_episode_reset(self):
        self._trace_anchors=[];self._trace_candidate=None;self._trace_goal=None
        self._trace_segments=[];self._trace_last_position=None;self._trace_path=0.
        self._trace_snapshot_goal=None

    def _reset(self):
        super()._reset();self._trace_episode+=1;self._trace_episode_reset()

    def _trace_identity(self):
        if not 0<=self._trace_episode<len(self._trace_sources):
            if self._trace_sources:raise RuntimeError('Trace episode order exceeded fixed dataset')
            return dict(episode_index=self._trace_episode)
        e=self._trace_sources[self._trace_episode]
        return dict(episode_index=self._trace_episode,scene_id=e['scene_id'],episode_id=str(e['episode_id']))

    def _trace_event(self,event,**values):
        row=dict(**self._trace_identity(),step=self._num_steps,event=event,
                 candidate_spatial_id=self._trace_candidate,**values)
        with (Path(os.environ['AV_RUN_DIR'])/'trace.jsonl').open('a') as f:
            f.write(json.dumps(row,default=native)+'\n')

    def _pre_step(self,observations,masks):
        super()._pre_step(observations,masks)
        self._trace_segments=[];self._trace_goal=None;self._trace_candidate=None
        p=self._observations_cache['robot_xy'].copy()
        if self._trace_last_position is not None:self._trace_path+=float(np.linalg.norm(p-self._trace_last_position))
        self._trace_last_position=p
        if self._num_steps==0:self._trace_event('episode_start',target=self._target_object)

    def _update_object_map(self,*args):
        self._trace_rgbd=args
        return super()._update_object_map(*args)

    def _get_target_object_location(self,position):
        goal=super()._get_target_object_location(position)
        self._trace_goal=None if goal is None else np.asarray(goal).copy()
        if goal is not None:
            distances=[np.linalg.norm(goal[:2]-p) for p in self._trace_anchors]
            if distances and min(distances)<=.75:self._trace_candidate=int(np.argmin(distances))
            else:self._trace_candidate=len(self._trace_anchors);self._trace_anchors.append(goal[:2].copy())
        return goal

    def _get_policy_info(self,detections):
        info=super()._get_policy_info(detections)
        cache=self._observations_cache
        self._trace_state=dict(robot_xy=cache['robot_xy'].copy(),heading=float(cache['robot_heading']),
            selected_target=self._trace_goal,nav_goal=np.asarray(info['nav_goal']).copy(),
            target_detected=bool(info['target_detected']),stop_called=bool(info['stop_called']),
            target=self._target_object,path_m=self._trace_path,segments=self._trace_segments,
            detections=detections.to_json(),verification_active=getattr(self,'_av_active',None) is not None)
        save=bool(self._trace_segments) and (self._num_steps%10==0 or self._trace_snapshot_goal is None
            or (self._trace_goal is not None and np.linalg.norm(self._trace_goal[:2]-self._trace_snapshot_goal)>.25)
            or info['stop_called'] or getattr(self,'_av_active',None) is not None)
        if save:
            import cv2
            folder=Path(os.environ['AV_RUN_DIR'])/'candidate_evidence';folder.mkdir(exist_ok=True)
            name=f'ep{self._trace_episode}_step{self._num_steps}'
            rgb,depth,tf,lo,hi,fx,fy=self._trace_rgbd
            if not cv2.imwrite(str(folder/(name+'.jpg')),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):
                raise RuntimeError('Failed to save trace RGB')
            np.savez_compressed(folder/(name+'.npz'),normalized_depth=depth,tf_camera_to_episodic=tf,
                min_depth=lo,max_depth=hi,fx=fx,fy=fy,object_mask=self._object_masks,
                target_point_cloud=info['target_point_cloud'])
            self._trace_state['evidence']='candidate_evidence/'+name
            if self._trace_goal is not None:self._trace_snapshot_goal=self._trace_goal[:2].copy()
        info.update(trace_episode_index=self._trace_episode,trace_path_m=self._trace_path)
        return info

    def act(self,*args,**kwargs):
        start=time.perf_counter();self._trace_state=None
        result=super().act(*args,**kwargs)
        state=self._trace_state or dict(incomplete_policy_state=True)
        self._trace_event('step',action=int(result.actions.item()),step_completed=max(0,self._num_steps-1),
                          elapsed_s=time.perf_counter()-start,**state)
        return result
