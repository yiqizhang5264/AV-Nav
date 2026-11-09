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
    def test_vqa_answer_parser_accepts_explicit_and_final_line_answers(self) -> None:
        patch = _patch_text("official_c9800d0_vqa_answer_parser.patch")
        added = _added_lines(patch)
        start = added.index("def parse_vqa_binary_answer(answer: str) -> str:")
        end = added.index(
            '    raise ValueError("VQA response has no unambiguous yes/no final answer")'
        )
        source = "\n".join(added[start : end + 1])
        namespace = {"re": re}
        exec(source, namespace)
        parse = namespace["parse_vqa_binary_answer"]

        self.assertEqual(parse("Answer: yes"), "yes")
        self.assertEqual(parse("**Answer:** No"), "no")
        self.assertEqual(
            parse("Reasoning: There is no bed in this dining room.\nNo"),
            "no",
        )
        self.assertEqual(parse("Some explanation.\n**YES**."), "yes")

    def test_vqa_answer_parser_rejects_ambiguous_reasoning(self) -> None:
        patch = _patch_text("official_c9800d0_vqa_answer_parser.patch")
        added = _added_lines(patch)
        start = added.index("def parse_vqa_binary_answer(answer: str) -> str:")
        end = added.index(
            '    raise ValueError("VQA response has no unambiguous yes/no final answer")'
        )
        source = "\n".join(added[start : end + 1])
        namespace = {"re": re}
        exec(source, namespace)
        parse = namespace["parse_vqa_binary_answer"]

        with self.assertRaises(ValueError):
            parse("There is no bed in the image, so the detection is incorrect.")
        with self.assertRaises(ValueError):
            parse("yes or no cannot be determined")

    def test_video_map_filters_out_of_bounds_points(self) -> None:
        patch = _patch_text("official_c9800d0_video_map_bounds.patch")
        added = "\n".join(_added_lines(patch))
        self.assertIn("valid_grid_xy = grid_xy[", added)
        self.assertIn("grid_xy[:, 0] < new_map.shape[0]", added)
        self.assertIn("grid_xy[:, 1] < new_map.shape[1]", added)
        self.assertIn("grid_xy[:, 0] >= 0", added)
        self.assertIn("grid_xy[:, 1] >= 0", added)
        self.assertIn(
            "new_map[valid_grid_xy[:, 0], valid_grid_xy[:, 1]]",
            added,
        )

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
