import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from scripts.room_frame_stream import atomic_json, frames, wait_file
from scripts.vlfm_evidence_adapter import create_vlfm_evidence_overlay


class OnlineRoomStreamTests(unittest.TestCase):
    def test_blocks_until_current_frame_and_stops_at_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrived = []
            worker = threading.Thread(target=lambda: arrived.extend(frames(root, True)))
            worker.start()
            time.sleep(.05)
            self.assertEqual(arrived, [])
            (root / '0000.npz').write_bytes(b'complete payload')
            atomic_json(root / '0000.ready.json', dict(frame=0, last=True))
            worker.join(timeout=1)
            self.assertFalse(worker.is_alive())
            self.assertEqual(arrived, [(0, root / '0000.npz', True)])
            self.assertFalse((root / '0001.npz').exists())

    def test_rejects_wrong_identity_or_incomplete_payload_and_times_out(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / '0000.ready.json', dict(frame=1, last=True))
            with self.assertRaisesRegex(ValueError, 'Out-of-order'):
                next(frames(root, True))
            atomic_json(root / '0000.ready.json', dict(frame=0, last=True))
            with self.assertRaisesRegex(ValueError, 'complete frame'):
                next(frames(root, True))
            with self.assertRaises(TimeoutError):
                wait_file(root / 'absent', timeout=.01)

    def test_online_hook_precedes_step_and_default_baseline_has_no_hook(self):
        root = Path(__file__).resolve().parents[1] / 'external/vlfm'
        for enabled in [False, True]:
            overlay = create_vlfm_evidence_overlay(root, room_online=enabled)
            self.addCleanup(overlay.cleanup)
            source = (Path(overlay.name) / 'vlfm/utils/vlfm_trainer.py').read_text()
            if enabled:
                self.assertLess(source.index('.observe(self.envs,'), source.index('outputs = self.envs.step(step_data)'))
                self.assertIn('.finish(episode_stats, failure_cause)', source)
            else:
                self.assertNotIn('get_online_rooms', source)


if __name__ == '__main__':
    unittest.main()
