"""Encode actual online snapshots without recomputing any segmentation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',required=True)
    p.add_argument('--output-dir',required=True)
    args=p.parse_args()
    import cv2
    root,output=Path(args.run_root),Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    for i in range(5):
        job=root/f'online_full5/case_{i:02d}'
        source=json.loads((job/'inputs/00/manifest.json').read_text())
        n=source['frames']
        images=sorted((job/'visualizations/00').glob('[0-9][0-9][0-9][0-9].png'))
        expected=sorted(set(list(range(9,n,10))+[n-1]))
        if [int(p.stem) for p in images]!=expected:
            raise ValueError('Online visualization sequence is incomplete')
        scene=Path(source['episode']['scene_id']).name.split('.')[0]
        first=cv2.imread(str(images[0]));height,width=first.shape[:2]
        video=output/f'{scene}_online.mp4'
        encoded=video.with_name(scene+'_frames_source.mp4') if shutil.which('ffmpeg') else video
        writer=cv2.VideoWriter(str(encoded),cv2.VideoWriter_fourcc(*'mp4v'),1.,(width,height))
        if not writer.isOpened():raise RuntimeError('Video writer unavailable')
        for image in images:
            data=cv2.imread(str(image))
            if data.shape!=first.shape:raise ValueError('Online snapshot dimensions changed')
            writer.write(data)
        writer.release()
        if encoded!=video:
            subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n','-i',str(encoded),
                            '-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],
                           stdin=subprocess.DEVNULL,check=True)
        check=cv2.VideoCapture(str(video));frames=int(check.get(cv2.CAP_PROP_FRAME_COUNT));check.release()
        if frames!=len(images):raise ValueError('Encoded frame count differs')
        records.append(dict(scene=scene,simulation_frames=n,video_frames=frames,fps=1,codec='h264' if encoded!=video else 'mp4v',
                            snapshot_steps=expected,video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                            snapshots_sha256={image.name:hashlib.sha256(image.read_bytes()).hexdigest() for image in images}))
    metadata=dict(generator_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
                  timing='One video frame per saved online visualization; snapshots every 10 simulation steps and terminal step; playback speed is not wall-clock speed',
                  segmentation_recomputed=False,records=records)
    (output/'video_manifest.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
