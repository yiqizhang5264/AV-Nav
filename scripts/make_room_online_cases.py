"""Derive single-episode online jobs from an already fixed scene sample."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    sample, output = Path(args.sample_dir), Path(args.output_dir)
    selection = json.loads((sample / 'selection.json').read_text())
    payload = (sample / 'episodes.json.gz').read_bytes()
    if hashlib.sha256(payload).hexdigest() != selection['sha256']:
        raise ValueError('Original sample hash mismatch')
    data = json.loads(gzip.decompress(payload))
    output.mkdir(parents=True, exist_ok=False)
    for index, (episode, case) in enumerate(zip(data['episodes'], selection['cases'])):
        folder = output / f'{index:02d}'
        folder.mkdir()
        key = Path(episode['scene_id']).name + '_' + episode['object_category']
        subset = dict(data, episodes=[episode], goals_by_category={key: data['goals_by_category'][key]})
        encoded = gzip.compress(json.dumps(subset, sort_keys=True).encode(), mtime=0)
        record = dict(selection, count=1, cases=[case], sha256=hashlib.sha256(encoded).hexdigest(),
                      parent_selection_sha256=selection['sha256'], parent_case_index=index,
                      execution_note='One independently seeded VLFM evaluation process; Habitat runtime ID is 0; no resampling')
        (folder / 'episodes.json.gz').write_bytes(encoded)
        (folder / 'selection.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
