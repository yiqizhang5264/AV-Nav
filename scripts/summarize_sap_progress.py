"""Read-only progress report; never label a partial evaluation as complete."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(root):
    root = Path(root)
    complete, active, failed = [], [], []
    rows = []
    for scene in sorted(p for p in root.iterdir() if p.is_dir()):
        attempts = sorted(scene.glob('attempt_*'))
        chosen = None
        for attempt in attempts:
            summary = attempt/'summary.json'
            if summary.exists():
                status = json.loads(summary.read_text())
                if status.get('complete'):
                    chosen = attempt
                    break
                failed.append(str(attempt.relative_to(root)))
        if chosen:
            complete.append(scene.name)
            rows.extend(json.loads(line) for line in (chosen/'episodes.jsonl').read_text().splitlines())
        elif attempts:
            current = attempts[-1]
            count = sum(1 for line in (current/'episodes.jsonl').read_text().splitlines() if line) if (current/'episodes.jsonl').exists() else 0
            active.append(dict(scene=scene.name, logged_episodes=count,
                               status='failed' if (current/'summary.json').exists() else 'unfinished'))
    result = dict(complete=(root/'summary.json').exists(), complete_scenes=len(complete),
                  complete_scene_episodes=len(rows), unfinished=active, failed_attempts=failed,
                  target_counts=dict(Counter(r['metrics'].get('target_object') for r in rows)))
    if rows:
        result['partial_metrics_completed_scenes_only'] = {
            k: sum(float(r['metrics'][k]) for r in rows)/len(rows) for k in ('success','spl')}
        result['zero_trigger_episodes'] = sum(r['metrics'].get('sap_triggers') == 0 for r in rows)
    if result['complete']:
        result['final_summary'] = json.loads((root/'summary.json').read_text())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2))
