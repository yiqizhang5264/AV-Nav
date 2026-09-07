"""Launch in upstream cwd, save provenance before running; never overwrite a run."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from av_nav.config import Config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--dataset", help="Absolute filtered split .json.gz")
    p.add_argument("--split", default="val")
    p.add_argument("--gpu", default="3")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {".",".."}:
        p.error("run-id must be a directory name")
    cfg_path = Path(args.config).resolve()
    cfg = Config.load(cfg_path)
    if args.dataset and not any(x in str(Path(args.dataset).resolve()).lower() for x in ('hm3d','mp3d')):
        p.error('Pinned VLFM infers dataset type from path: include hm3d or mp3d in the split filename')
    run = ROOT/"runs"/args.run_id
    if run.exists():
        p.error(f"Run exists: {run}")
    upstream = ROOT/"external"/"vlfm"
    env = os.environ.copy()
    env.update(AV_CONFIG=str(cfg_path), AV_RUN_DIR=str(run), AV_STRATEGY=cfg.strategy,
               CUDA_VISIBLE_DEVICES=args.gpu, PYTHONHASHSEED=str(cfg.seed),
               PYTHONPATH=os.pathsep.join([str(ROOT),str(upstream),env.get("PYTHONPATH","")]))
    env.update({k:env.get(k,str(v)) for k,v in dict(GROUNDING_DINO_PORT=13181,BLIP2ITM_PORT=13182,SAM_PORT=13183,YOLOV7_PORT=13184).items()})
    command = [sys.executable,"-u","-m","av_nav.runtime",
               f"habitat.seed={cfg.seed}", f"habitat_baselines.test_episode_count={args.episodes}",
               f"habitat.environment.max_episode_steps={args.max_steps}",
               f"habitat_baselines.eval.split={args.split}",
               "habitat.environment.iterator_options.shuffle=False",
               "habitat_baselines.eval.video_option=[]",
               f"habitat_baselines.tensorboard_dir={run/'tb'}", f"hydra.run.dir={run/'hydra'}"]
    if cfg.strategy != "baseline":
        command += ["habitat_baselines.rl.policy.name=AVHabitatPolicy"]
    if args.dataset:
        command += [f"habitat.dataset.data_path={Path(args.dataset).resolve()}"]
    metadata = dict(config=asdict(cfg), command=command, gpu=args.gpu,
                    started=datetime.now(timezone.utc).isoformat(),
                    av_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                    upstream_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=upstream,text=True).strip(),
                    dirty=subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True),
                    dataset_sha256=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest() if args.dataset else None)
    print(json.dumps(metadata,indent=2))
    if args.dry_run:
        return
    if not (upstream/"data"/"dummy_policy.pth").exists():
        p.error("Missing upstream data/dummy_policy.pth: run bootstrap_server.sh first")
    run.mkdir(parents=True)
    (run/"manifest.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    freeze = subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True)
    (run/"environment.txt").write_text(freeze.stdout,encoding="utf-8")
    with (run/"console.log").open("w",encoding="utf-8") as f:
        result = subprocess.run(command,cwd=upstream,env=env,stdout=f,stderr=subprocess.STDOUT)
    (run/"exit.json").write_text(json.dumps(dict(returncode=result.returncode)),encoding="utf-8")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
