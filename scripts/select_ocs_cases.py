"""Freeze a stratified OCS diagnostic cohort without looking at intervention outputs.

Uses historical HM3Dv1/v2 caches, not train calibration or an efficacy cohort.
The server verification is read-only and checks inputs, exact console modes and
episode identity. Full baseline decision reconstruction remains a later gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


DATASETS = ("hm3dv1", "hm3dv2")
STRATA = ("strict_low_yield_failure", "other_audited_failure", "success_control")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return row["dataset"], row["episode_key"]


def ranking_digest(row, seed=20261030):
    scene = row.get("scene") or row["episode_key"].split(".basis.glb_")[0]
    canonical = f"{seed}/{row['dataset']}/val/{scene}/{row['episode_key']}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def candidates(failures, progress, terminals):
    """Require completed historical audit for failure controls, never absence of data."""
    audited = {}
    for row in progress["cases"]:
        k = key(row)
        if k in audited:
            raise ValueError(f"Duplicate historical audit identity: {k}")
        audited[k] = row
    groups = {name: [] for name in STRATA}
    seen = set()
    for row in failures["episodes"]:
        if row["dataset"] not in DATASETS:
            continue
        k = key(row)
        if k in seen:
            raise ValueError(f"Duplicate failure identity: {k}")
        seen.add(k)
        audit = audited.get(k)
        if (not row.get("strict_2d_input_eligible") or not audit
                or audit.get("status") != "analyzed"):
            continue
        category = audit.get("episode_category")
        if category == "contains_low_yield_repeated_return":
            stratum = STRATA[0]
        elif category in ("spatial_returns_without_confirmed_low_yield_pattern", "no_qualifying_spatial_return"):
            stratum = STRATA[1]
        else:
            continue
        record = dict(row)
        record.update(ocs_stratum=stratum, split="val", audit_category=category)
        groups[stratum].append(record)
    for dataset, terminal in terminals.items():
        for row in terminal["episodes"]:
            ev = row.get("evidence", {})
            if not (row.get("success") is True and row.get("console_join_valid")
                    and row.get("stairs") in (0, 0.0)
                    and ev.get("contiguous_from_zero")
                    and len(ev.get("modes", [])) == row.get("steps_recorded")
                    and "explore" in ev.get("modes", [])):
                continue
            record = {
                "dataset": dataset, "split": "val", "scene": row["scene"],
                "episode_id": str(row["episode_id"]),
                "episode_key": f"{row['scene']}.basis.glb_{row['episode_id']}",
                "ocs_stratum": STRATA[2], "original_success": True,
                "original_steps_recorded": row["steps_recorded"],
                "original_mode_sequence": ev["modes"], "original_mode_counts": ev["mode_counts"],
                "reported_stairs": row["stairs"], "trace_source": row["step_trace"],
                "console_source": row["console"], "console_episode_end_line": ev["episode_end_line"],
                "terminal_class": "success", "input_validation_pending": True,
            }
            if key(record) in seen:
                raise ValueError(f"Outcome conflict or duplicate success identity: {key(record)}")
            seen.add(key(record))
            groups[STRATA[2]].append(record)
    return groups


def select(groups, n=4, seed=20261030):
    if n < 1:
        raise ValueError("per-stratum must be positive")
    selected, counts = [], {}
    for stratum in STRATA:
        ordered = sorted(groups[stratum], key=lambda r: (ranking_digest(r, seed), key(r)))
        counts[stratum] = len(ordered)
        if len(ordered) < n:
            raise ValueError(f"Insufficient {stratum}: {len(ordered)} < {n}; do not silently relax")
        for rank, record in enumerate(ordered[:n]):
            row = dict(record)
            row.update(selection_rank_in_stratum=rank, selection_sha256=ranking_digest(row, seed),
                       selection_seed=seed, selection_no_intervention_outcome=True)
            selected.append(row)
    # First three cases contain one per stratum for initial smoke; then round-robin.
    selected.sort(key=lambda r: (r["selection_rank_in_stratum"], STRATA.index(r["ocs_stratum"])))
    return selected, counts


def verify_remote(records, host, remote_repo):
    code = r'''
import hashlib, json, pathlib, sys
from collections import Counter
import yaml
sys.path.insert(0, REMOTE_REPO + '/scripts')
from build_allfail_manifest import parse_console, scene_name
records = json.loads(PAYLOAD)
console_cache = {}
out = []
for r in records:
    p = pathlib.Path(r['trace_source']); epdir = p.parent; attempt = epdir.parents[2]
    raw = p.read_bytes(); rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
    digest = hashlib.sha256(raw).hexdigest()
    if r.get('source_trace_sha256') and digest != r['source_trace_sha256']:
        raise RuntimeError('Historical trace changed: ' + r['episode_key'])
    n = r['original_steps_recorded']
    if len(rows) != n or [x['step'] for x in rows] != list(range(n)) or not rows[-1]['done']:
        raise RuntimeError('Invalid completed trace: ' + r['episode_key'])
    meta = json.loads((epdir/'episode.json').read_text())
    result = json.loads((epdir/'result.json').read_text())
    if (scene_name(meta['scene_id']) != r['scene'] or str(meta['episode_id']) != r['episode_id']
            or bool(result['metrics']['success']) != r['original_success']):
        raise RuntimeError('Identity or outcome mismatch: ' + r['episode_key'])
    cp = r['console_source']
    if cp not in console_cache: console_cache[cp] = parse_console(pathlib.Path(cp))
    c = console_cache[cp].get((r['scene'], r['episode_id']))
    if (not c or not c['contiguous'] or c['repeated_key'] or c['modes'] != r['original_mode_sequence']
            or c['actions'] != [x['action'] for x in rows]):
        raise RuntimeError('Exact mode/action join failed: ' + r['episode_key'])
    config_path = attempt/'hydra/.hydra/config.yaml'; cfg = yaml.safe_load(config_path.read_text())
    pc = cfg['habitat_baselines']['rl']['policy']
    if pc['name'] not in ('HabitatITMPolicyV2','HabitatITMPolicyV3'):
        raise RuntimeError('Unsupported policy')
    for folder, suffix in (('rgb','.png'),('depth','.npy'),('sensors','.npz')):
        actual = {int(f.stem) for f in (epdir/folder).glob('*'+suffix) if f.stem.isdigit()}
        if set(range(n)) - actual: raise RuntimeError('Missing '+folder+': '+r['episode_key'])
    channels = len(pc['text_prompt'].split('|')); events = Counter()
    for ev in (epdir/'model_events').glob('*_blip2_itm'):
        if (ev/'event.json').is_file() and (ev/'image.png').is_file(): events[int(ev.name.split('_')[0])] += 1
    if any(events[s] != channels for s in range(n)):
        raise RuntimeError('ITM cache count mismatch: '+r['episode_key'])
    # Identity metadata contains no policy oracle. Preserve exact source task hash separately.
    sha = lambda q: hashlib.sha256(q.read_bytes()).hexdigest()
    out.append(dict(dataset=r['dataset'],episode_key=r['episode_key'],source_trace_sha256=digest,
        source_episode_meta_sha256=sha(epdir/'episode.json'),source_result_sha256=sha(epdir/'result.json'),
        config_source=str(config_path),source_config_sha256=sha(config_path),
        input_validation_pending=False,input_replay_eligible=True,strict_2d_input_eligible=True,
        task_recorded_scene_id=meta['scene_id'],
        task_recorded_start_position=meta.get('start_position'),task_recorded_start_rotation=meta.get('start_rotation'),
        recorded_episode_metadata=meta,original_category=rows[0].get('policy_info',{}).get('target_object'),
        cached_itm_event_count=sum(events.values()),cached_itm_expected_channels=channels,
        task_index_verified=True))
print(json.dumps(out))
'''.replace("REMOTE_REPO", repr(remote_repo)).replace("PAYLOAD", repr(json.dumps(records)))
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
         "/home/zyq/miniconda3/envs/vlfm/bin/python", "-"],
        input=code, text=True, encoding="utf-8", capture_output=True, timeout=180, check=True)
    verified = {key(r): r for r in json.loads(completed.stdout)}
    if set(verified) != {key(r) for r in records}:
        raise ValueError("Remote returned a different cohort")
    return [dict(r, **{k: v for k, v in verified[key(r)].items() if k not in ('dataset','episode_key')}) for r in records]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--failure-manifest", type=Path, required=True)
    p.add_argument("--progress", type=Path, required=True)
    p.add_argument("--terminal-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-stratum", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261030)
    p.add_argument("--ssh-host", default="yingu")
    p.add_argument("--remote-repo", default="/home/zyq/AV-Nav-worktrees/vlfm-allfail-audit-07d249a")
    p.add_argument("--dry-run", action="store_true", help="No remote verification; not executable")
    args = p.parse_args()
    if args.output.exists(): raise SystemExit("Refusing to overwrite frozen selection")
    terminals = {d: read_json(args.terminal_dir/f"vlfm_{d}_terminal.json") for d in DATASETS}
    records, counts = select(candidates(read_json(args.failure_manifest), read_json(args.progress), terminals),
                             args.per_stratum, args.seed)
    if not args.dry_run: records = verify_remote(records, args.ssh_host, args.remote_repo)
    sources = [args.failure_manifest, args.progress, *[args.terminal_dir/f"vlfm_{d}_terminal.json" for d in DATASETS]]
    doc = dict(schema="vlfm.ocs_diagnostic_selection.v1", created_utc=datetime.now(timezone.utc).isoformat(),
               seed=args.seed, source_sha256={str(x): file_sha(x) for x in sources},
               selector_sha256=file_sha(__file__), eligible_counts=counts, episode_count=len(records),
               stratum_counts=dict(Counter(r['ocs_stratum'] for r in records)),
               executable_input_manifest=not args.dry_run, first_smoke_case_indices=[0,1,2],
               no_intervention_outcome_used=True, split="val", purpose="frozen-input mechanism diagnosis only",
               no_train_calibration=True, no_new_navigation_rollout=True,
               no_replacement_on_cache_failure=True, episodes=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2)+'\n', encoding="utf-8")
    print(json.dumps({"output":str(args.output),"sha256":file_sha(args.output),"eligible_counts":counts,
                      "cases":[{"index":i,"dataset":r['dataset'],"episode_key":r['episode_key'],"stratum":r['ocs_stratum']}
                               for i,r in enumerate(records)]}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
