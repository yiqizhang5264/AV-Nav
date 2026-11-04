import re
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "adapters" / "beliefmapnav"


def _patch_text(name: str) -> str:
    return (PATCH_DIR / name).read_text(encoding="utf-8")


def _added_lines(patch: str) -> list[str]:
    return [
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


class BeliefMapNavOfficialFixTests(unittest.TestCase):
    def test_complete_split_covers_every_episode_once(self) -> None:
        patch = _patch_text("official_c9800d0_complete_split.patch")
        starts_match = re.search(r"^ start_splits=\(([^)]+)\)$", patch, re.MULTILINE)
        ends_match = re.search(r"^\+end_splits=\(([^)]+)\)$", patch, re.MULTILINE)
        self.assertIsNotNone(starts_match)
        self.assertIsNotNone(ends_match)

        starts = [int(value) for value in starts_match.group(1).split()]
        ends = [int(value) for value in ends_match.group(1).split()]
        self.assertEqual(len(starts), 8)
        self.assertEqual(len(ends), 8)
        self.assertEqual(starts[0], 0)
        self.assertEqual(ends[-1], 2000)
        self.assertTrue(all(starts[index] == ends[index - 1] for index in range(1, 8)))

        covered = [episode for start, end in zip(starts, ends) for episode in range(start, end)]
        self.assertEqual(covered, list(range(2000)))

    def test_initialization_finishes_on_twelfth_turn(self) -> None:
        patch = _patch_text("official_c9800d0_initialization_12_turns.patch")
        added = _added_lines(patch)
        start = added.index("        self._initialize_step += 1")
        self.assertIn("             self._done_initializing = True", patch)
        logic = added[start : start + 2] + [
            "            self._done_initializing = True",
            "            self._obstacle_map._tight_search_thresh = False",
        ]
        source = "def initialize(self):\n" + textwrap.indent(
            textwrap.dedent("\n".join(logic)), "    "
        )
        namespace: dict[str, object] = {}
        exec(source, namespace)

        obstacle_map = type("ObstacleMap", (), {"_tight_search_thresh": True})()
        policy = type(
            "Policy",
            (),
            {
                "_initialize_step": 0,
                "_done_initializing": False,
                "_obstacle_map": obstacle_map,
            },
        )()
        initialize = namespace["initialize"]
        for _ in range(11):
            initialize(policy)
            self.assertFalse(policy._done_initializing)
        initialize(policy)
        self.assertEqual(policy._initialize_step, 12)
        self.assertTrue(policy._done_initializing)
        self.assertFalse(policy._obstacle_map._tight_search_thresh)

    def test_unknown_category_has_dict_schema_and_normalized_probabilities(self) -> None:
        patch = _patch_text("official_c9800d0_unknown_category_schema.patch")
        added = _added_lines(patch)
        start = added.index(
            "        if not isinstance(attributes_list, list) or len(attributes_list) != 3:"
        )
        logic = added[start:]
        end = logic.index("        }") + 1
        source = "def normalize(attributes_list, target):\n" + textwrap.indent(
            textwrap.dedent("\n".join(logic[:end])), "    "
        )
        namespace: dict[str, object] = {}
        exec(source, namespace)
        normalize = namespace["normalize"]

        result = normalize(
            [
                [" garage ", "workshop", "utility room"],
                ["near wall", "on shelf", "by door"],
                ["toolbox", "workbench", "cabinet"],
            ],
            "drill",
        )
        self.assertEqual(set(result), {"attributes", "probability"})
        self.assertEqual(result["attributes"][-1], ["drill"])
        self.assertEqual(result["probability"][-1], [1.0])
        self.assertTrue(
            all(abs(sum(level) - 1.0) < 1e-9 for level in result["probability"])
        )
        self.assertNotIn(
            "            objects_attributes += [[self._target_object]]",
            _added_lines(patch),
        )

        with self.assertRaises(ValueError):
            normalize([["garage"]], "drill")
        with self.assertRaises(ValueError):
            normalize([["garage"], [], ["toolbox"]], "drill")


if __name__ == "__main__":
    unittest.main()
