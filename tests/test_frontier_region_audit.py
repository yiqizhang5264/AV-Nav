import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from frontier_region_audit import AuditConfig, _local_connection, _merge_intervals, analyze_episode


def fixture(extra_gain=0, different_frontier=False, semantic=False, terminal="nf"):
    # Walk 5 m out and return, then wait at the old region until an empty-frontier STOP.
    positions = [[float(x), 0.0] for x in np.arange(0, 5.01, 0.25)]
    positions += [[float(x), 0.0] for x in np.arange(4.75, -0.01, -0.25)]
    positions += [[0.0, 0.0]] * 12
    rows, source = [], []
    for step, position in enumerate(positions):
        goal = [position[0] + (0.3 if different_frontier else 0), 0.0]
        row = dict(step=step, robot_xy=position, mode="explore", original_action=1,
                   n_frontiers=2, normalization_pixels_per_meter=10,
                   new_cumulative_seen_cells=0, replayed_nav_goal=goal,
                   decision={"selected_frontier_xy": goal, "branch": "highest_noncyclic"})
        rows.append(row)
        source.append({"step": step, "policy_info": {"target_detected": semantic and step == 40}})
    rows[42]["new_cumulative_seen_cells"] = int(extra_gain * 100)
    if terminal == "nf":
        rows[-1].update(n_frontiers=0, original_action=0)
        rows[-1]["decision"] = {"branch": "no_frontiers_stop", "event": "empty_frontier_stop"}
    free = np.ones((240, 240), dtype=bool)
    context = dict(navigable=free, ever_explored=free, obstacle=np.zeros_like(free),
                   pixels_per_meter=10, episode_pixel_origin=[120, 120])
    return rows, source, context


class RegionAuditTests(unittest.TestCase):
    def test_upstream_zero_sentinel_is_empty_frontier_branch(self):
        rows, source, context = fixture()
        rows[-1]["n_frontiers"] = 1
        result = analyze_episode(rows, source, map_context=context)
        self.assertTrue(result["events"][-1]["followup_terminal_empty_frontier"])

    def test_low_yield_return_and_conserved_cost(self):
        rows, source, context = fixture()
        result = analyze_episode(rows, source, map_context=context)
        self.assertGreater(result["spatial_return_events"], 0)
        self.assertGreater(result["low_yield_return_events"], 0)
        self.assertLessEqual(result["low_yield_union_observed_actions"], len(rows) - 1)
        self.assertLessEqual(result["low_yield_union_path_m"], result["observed_path_m"])
        self.assertTrue(all(event["repeated_route_fraction"] > 0.99 for event in result["events"]))

    def test_new_goal_identity_does_not_prevent_detection(self):
        rows, source, context = fixture(different_frontier=True)
        result = analyze_episode(rows, source, map_context=context)
        self.assertGreater(result["low_yield_return_events"], 0)
        self.assertFalse(result["events"][0]["same_frontier_identity_required"])

    def test_destination_gain_not_only_transport_gain(self):
        rows, source, context = fixture(extra_gain=3.0)
        result = analyze_episode(rows, source, map_context=context)
        # At least the final return must include that later area gain.
        self.assertTrue(any(event["category"] == "observed_area_gain_return" for event in result["events"]))

    def test_missing_semantics_or_map_is_not_confirmed(self):
        rows, source, context = fixture()
        self.assertEqual(analyze_episode(rows, source)["low_yield_return_events"], 0)
        self.assertEqual(analyze_episode(rows, map_context=context)["low_yield_return_events"], 0)

    def test_source_records_without_target_fields_are_missing(self):
        rows, source, context = fixture()
        source = [{"step": row["step"]} for row in source]
        result = analyze_episode(rows, source, map_context=context)
        self.assertEqual(result["low_yield_return_events"], 0)
        self.assertTrue(result["events"][0]["semantic_evidence"]["missing_target_detection_field_steps"])

    def test_single_frontier_is_potentially_necessary(self):
        rows, source, context = fixture()
        for row in rows:
            if row["n_frontiers"]:
                row["n_frontiers"] = 1
        result = analyze_episode(rows, source, map_context=context)
        self.assertEqual(result["low_yield_return_events"], 0)
        self.assertTrue(any(event["category"] == "possibly_necessary_retreat_return" for event in result["events"]))

    def test_semantic_purpose_is_not_low_yield(self):
        rows, source, context = fixture(semantic=True)
        result = analyze_episode(rows, source, map_context=context)
        self.assertTrue(any(event["category"] == "target_evidence_or_target_navigation_return" for event in result["events"]))

    def test_wall_guard_and_unknown_guard(self):
        rows, source, context = fixture()
        context["navigable"][:, 123] = False
        connection = _local_connection([0.2, 0], [0.4, 0], context, AuditConfig())
        self.assertEqual(connection["status"], "unresolved")
        context["ever_explored"][:] = False
        self.assertEqual(_local_connection([0, 0], [0, 0], context, AuditConfig())["status"], "unresolved")

    def test_truncated_destination_not_called_low_yield(self):
        rows, source, context = fixture(terminal="budget")
        result = analyze_episode(rows, source, map_context=context)
        terminal = result["events"][-1]
        self.assertFalse(terminal["followup_complete"])
        self.assertEqual(terminal["category"], "insufficient_evidence_return")

    def test_pose_gap_and_episode_reset_not_bridged(self):
        rows, source, context = fixture()
        rows[5]["robot_xy"] = [20, 0]
        self.assertEqual(analyze_episode(rows, source)["status"], "insufficient_evidence")
        rows, source, context = fixture()
        rows[20]["step"] = 0
        self.assertEqual(analyze_episode(rows)["status"], "insufficient_evidence")
        self.assertEqual(analyze_episode([])["status"], "insufficient_evidence")

    def test_no_return_and_state_isolation(self):
        rows, source, context = fixture()
        analyze_episode(rows, source, map_context=context)
        straight = rows[:15]
        result = analyze_episode(straight, source[:15], map_context=context)
        self.assertEqual(result["spatial_return_events"], 0)

    def test_union_intervals(self):
        self.assertEqual(_merge_intervals([(3, 8), (5, 10), (10, 12), (0, 2)]), [[0, 2], [3, 12]])


if __name__ == "__main__":
    unittest.main()
