"""Blocking per-step observation stream; future observations are unavailable."""
import json
from pathlib import Path
import time


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def wait_file(path, timeout=1800):
    deadline = time.monotonic() + timeout
    while not Path(path).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Timed out waiting for {path}')
        time.sleep(.02)
    return Path(path)


def frames(case, streaming=False):
    case = Path(case)
    if not streaming:
        paths = sorted(case.glob('[0-9][0-9][0-9][0-9].npz'))
        for index, path in enumerate(paths):
            yield index, path, index == len(paths) - 1
        return
    index = 0
    while True:
        ready = json.loads(wait_file(case / f'{index:04d}.ready.json').read_text())
        if ready['frame'] != index:
            raise ValueError('Out-of-order online frame')
        path = case / f'{index:04d}.npz'
        if not path.exists():
            raise ValueError('Ready marker precedes complete frame')
        yield index, path, ready['last']
        if ready['last']:
            return
        index += 1


def acknowledge(output, index, record):
    atomic_json(Path(output) / f'ack_{index:04d}.json',
                dict(frame=index, completed_ns=time.time_ns(), record=record))
