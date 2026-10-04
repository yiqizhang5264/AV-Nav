"""Strict paired results from two completed HM3Dv1 suites; no partial efficacy table."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def load_suite(root):
    root = Path(root)
    summary = json.loads((root/'summary.json').read_text())
    identity = json.loads((root/'suite.json').read_text())
    if summary.get('complete') is not True or summary.get('episodes') != 2000:
        raise ValueError('A complete 2000-episode HM3Dv1 suite is required')
    records = {}
    for line in (root/'episodes.jsonl').read_text().splitlines():
        row = json.loads(line)
        key = row['source_uid']
        expected_uid = row['source_sha256']+':'+str(row['source_row'])
        if key != expected_uid or key in records:
            raise ValueError('Invalid or duplicate immutable source identity')
        for name in ('success', 'spl'):
            value = row['metrics'][name]
            if not isinstance(value, (int,float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid metric: '+name)
        if row['metrics']['success'] not in (0,1):
            raise ValueError('Success must be binary')
        records[key] = row
    if len(records) != summary['episodes'] or len({r['scene_id'] for r in records.values()}) != 20:
        raise ValueError('Incorrect episode/scene coverage')
    for name in ('success','spl'):
        measured = sum(r['metrics'][name] for r in records.values())/len(records)
        if not math.isclose(measured, summary['metrics'][name], abs_tol=1e-12):
            raise ValueError('Aggregate does not match episode records')
    return identity, records


def pair_records(baseline, sap):
    if set(baseline) != set(sap) or not baseline:
        raise ValueError('Paired comparison requires identical nonempty source-row sets')
    groups = defaultdict(list)
    for key in sorted(sap):
        a,b = baseline[key],sap[key]
        for field in ('scene_id','episode_sha256','source_file','source_row','episode_id'):
            if a[field] != b[field]:
                raise ValueError('Mismatched source episode metadata: '+field)
        if a['metrics']['target_object'] != b['metrics']['target_object']:
            raise ValueError('Target mismatch')
        groups[b['metrics']['target_object']].append((a,b))

    def aggregate(pairs):
        n=len(pairs)
        return dict(episodes=n,
            baseline={k:sum(a['metrics'][k] for a,b in pairs)/n for k in ('success','spl')},
            sap={k:sum(b['metrics'][k] for a,b in pairs)/n for k in ('success','spl')},
            delta={k:sum(b['metrics'][k]-a['metrics'][k] for a,b in pairs)/n for k in ('success','spl')},
            recovered=sum(a['metrics']['success']==0 and b['metrics']['success']==1 for a,b in pairs),
            regressed=sum(a['metrics']['success']==1 and b['metrics']['success']==0 for a,b in pairs),
            zero_trigger_episodes=sum(b['metrics']['sap_triggers']==0 for a,b in pairs),
            vlm_calls=sum(b['metrics']['sap_vlm_calls'] for a,b in pairs),
            reposition_attempts=sum(b['metrics']['sap_repositions'] for a,b in pairs))
    return dict(overall=aggregate([pair for values in groups.values() for pair in values]),
                by_category={key:aggregate(values) for key,values in sorted(groups.items())})


def compare(baseline_root, sap_root):
    base_identity, baseline = load_suite(baseline_root)
    sap_identity, sap = load_suite(sap_root)
    if base_identity['variant'] != 'baseline' or sap_identity['variant'] != 'sap':
        raise ValueError('Expected baseline and SAP suites in that order')
    for key in ('source_sha256','config_sha256','scenes_dir'):
        if base_identity[key] != sap_identity[key]:
            raise ValueError('Suite identity mismatch: '+key)
    return dict(benchmark='HM3Dv1 ObjectNav val', complete=True,
                baseline_identity=base_identity, sap_identity=sap_identity,
                **pair_records(baseline,sap))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline', required=True)
    p.add_argument('--sap', required=True)
    p.add_argument('--output', required=True, help='New directory; never overwritten')
    args=p.parse_args()
    result=compare(args.baseline,args.sap)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    (out/'comparison.json').write_text(json.dumps(result,indent=2))
    lines=['# HM3Dv1 val: VLFM versus SAP category reproduction','',
           'Qwen3.5-9B thinking. Complete source-row paired evaluation; no tuning on val.','',
           '| Category | Episodes | VLFM SR | SAP SR | VLFM SPL | SAP SPL |',
           '|---|---:|---:|---:|---:|---:|']
    for category,row in [('All',result['overall']),*result['by_category'].items()]:
        a,b=row['baseline'],row['sap']
        lines.append(f"| {category} | {row['episodes']} | {a['success']:.4f} | {b['success']:.4f} | {a['spl']:.4f} | {b['spl']:.4f} |")
    row=result['overall']
    lines += ['',f"Recovered: {row['recovered']}; regressed: {row['regressed']}; zero-trigger episodes: {row['zero_trigger_episodes']}.",
              '', 'See comparison.json for exact suite identities and verification counts.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result['overall'],indent=2))


if __name__ == '__main__':
    main()
