"""Read-only decision instrumentation for pinned original VLFM 584ed56.

No policy rule is reimplemented.  AST insertions observe the locals produced by
the original method, after its real branches execute.  In particular, the
all-cyclic fallback's max-distance/original-index behavior is *not* repaired.
Install only in an isolated diagnostic process, before starting episodes.
"""

import ast
import copy
import functools
import hashlib
import inspect
import math
from pathlib import Path
import textwrap
import time


EXPECTED_SOURCE_SHA256 = "a910ecb6ff7282c62116de9a33f7a73434833ed06c0cbd47f8e0cb8384e794ea"
UPSTREAM_COMMIT = "584ed56008754fde7997d904983607def8328322"
_GLOBAL = "__vlfm_frontier_probe_recorder_20261010"
_PENDING = "_vlfm_frontier_probe_pending_20261010"


def _jsonable(value):
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        # Strict JSON: preserve initial -inf explicitly, never emit NaN literals.
        return {"nonfinite": repr(value)}
    if isinstance(value, (str, float, int, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError("Unsupported trace value: " + type(value).__name__)


class _Recorder:
    def __init__(self, emit, context_fn):
        self.emit = emit
        self.context_fn = context_fn

    def _context(self, policy, observations):
        return _jsonable(self.context_fn(policy, observations)) if self.context_fn else {}

    def begin(self, policy, observations, frontiers):
        if hasattr(policy, _PENDING):
            raise RuntimeError("Nested or unfinished frontier decision trace")
        record = {
            "schema": "vlfm.frontier_decision.v1",
            "event": "frontier_decision",
            "upstream_commit": UPSTREAM_COMMIT,
            "source_sha256": EXPECTED_SOURCE_SHA256,
            "context": self._context(policy, observations),
            "input_frontiers_xy": _jsonable(frontiers),
            "robot_xy": _jsonable(policy._observations_cache["robot_xy"]),
            "last_frontier_before": _jsonable(policy._last_frontier),
            "last_value_before": _jsonable(policy._last_value),
            "cyclic_checks": [],
            "branch": None,
            "started_monotonic_ns": time.monotonic_ns(),
        }
        setattr(policy, _PENDING, record)

    def sorted_snapshot(self, policy, sorted_pts, sorted_values):
        record = getattr(policy, _PENDING)
        record["sorted_frontiers_xy"] = _jsonable(sorted_pts)
        record["sorted_values"] = _jsonable(sorted_values)
        record["top_two_values"] = _jsonable(sorted_values[:2])
        # Coordinates may repeat: retain *all* matching original indices instead
        # of inventing a unique frontier ID or relying on array order stability.
        record["sorted_to_input_matches"] = [
            [i for i, point in enumerate(record["input_frontiers_xy"]) if point == sorted_point]
            for sorted_point in record["sorted_frontiers_xy"]
        ]

    def reduction(self, policy, values, mode):
        # Observe the actual V3 reduce function's input *after* it chose its
        # global branch; do not resample ValueMap or reevaluate reduction.
        record = getattr(policy, _PENDING, None)
        if record is None:
            return  # _reduce_values can also be invoked outside a decision.
        record["v3_reduction"] = {
            "input_order_target_exploration_values": _jsonable(values),
            "global_mode": mode,
            "exploration_threshold": _jsonable(policy._exploration_thresh),
            "rule": "max(target channel) < exploration_threshold selects exploration for all frontiers; otherwise target for all",
        }

    def cyclic(self, policy, index, frontier, cyclic, top_two_values):
        getattr(policy, _PENDING)["cyclic_checks"].append({
            "sorted_index": index,
            "frontier_xy": _jsonable(frontier),
            "cyclic": _jsonable(cyclic),
            "top_two_values": _jsonable(top_two_values),
        })

    def branch(self, policy, name, state):
        record = getattr(policy, _PENDING)
        record["branch"] = name
        if name == "keep_last":
            record["keep_last"] = {
                "sorted_index": state.get("curr_index"),
                "current_value": _jsonable(state.get("curr_value")),
                "rule": "current_value + 0.01 > last_value_before",
                "exact_or_near": "exact" if "closest_index" not in state else "within_0.5m",
            }
        if name == "all_cyclic_fallback":
            record["fallback_actual_rule"] = (
                "max original-input Euclidean-distance index reused as sorted-list index; "
                "recorded unchanged, not a closest-frontier policy"
            )

    def finish(self, policy, state):
        record = getattr(policy, _PENDING)
        index = int(state["best_frontier_idx"])
        record.update({
            "selected_sorted_index": index,
            "selected_input_indices": record["sorted_to_input_matches"][index],
            "selected_frontier_xy": _jsonable(state["best_frontier"]),
            "selected_value": _jsonable(state["best_value"]),
            "last_frontier_after": _jsonable(policy._last_frontier),
            "last_value_after": _jsonable(policy._last_value),
            "decision_elapsed_ms": (time.monotonic_ns() - record["started_monotonic_ns"]) / 1e6,
            "check_coverage": "only checks actually executed; untested candidates are not assumed noncyclic",
        })
        if record["branch"] is None:
            raise RuntimeError("Instrumentation did not observe a selection branch")
        delattr(policy, _PENDING)
        self.emit(_jsonable(record))

    def empty_stop(self, policy, observations, frontiers):
        self.emit({
            "schema": "vlfm.frontier_decision.v1",
            "event": "empty_frontier_stop",
            "upstream_commit": UPSTREAM_COMMIT,
            "source_sha256": EXPECTED_SOURCE_SHA256,
            "context": self._context(policy, observations),
            "input_frontiers_xy": _jsonable(frontiers),
            "robot_xy": _jsonable(policy._observations_cache.get("robot_xy")),
            "branch": "no_frontiers_stop",
            "sentinel_or_empty": "empty" if len(frontiers) == 0 else "zero_sentinel",
        })


def _call(method, arguments):
    return ast.parse(f"{_GLOBAL}.{method}({arguments})").body[0]


class _Instrument(ast.NodeTransformer):
    def __init__(self, kind):
        self.kind = kind
        self.counts = {"sort": 0, "cyclic": 0, "keep": 0, "rank": 0, "fallback": 0, "finish": 0, "stop": 0,
                       "reduce_explore": 0, "reduce_target": 0}

    def visit_Assign(self, node):
        self.generic_visit(node)
        if self.kind != "_get_best_frontier":
            return node
        target = ast.unparse(node.targets[0])
        value = ast.unparse(node.value)
        extra = None
        if target == "(sorted_pts, sorted_values)":
            if value != "self._sort_frontiers_by_value(observations, frontiers)":
                raise RuntimeError("Unexpected sorting anchor")
            self.counts["sort"] += 1
            extra = _call("sorted_snapshot", "self, sorted_pts, sorted_values")
        elif target == "cyclic":
            if value != "self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)":
                raise RuntimeError("Unexpected acyclic anchor")
            self.counts["cyclic"] += 1
            extra = _call("cyclic", "self, idx, frontier, cyclic, top_two_values")
        elif target == "best_frontier_idx" and value == "curr_index":
            self.counts["keep"] += 1
            extra = _call("branch", "self, 'keep_last', locals()")
        elif target == "best_frontier_idx" and value == "idx":
            self.counts["rank"] += 1
            extra = _call("branch", "self, 'highest_noncyclic', locals()")
        elif target == "best_frontier_idx" and isinstance(node.value, ast.Call):
            if not isinstance(node.value.func, ast.Name) or node.value.func.id != "max":
                raise RuntimeError("Unexpected fallback anchor")
            self.counts["fallback"] += 1
            extra = _call("branch", "self, 'all_cyclic_fallback', locals()")
        return [node, ast.copy_location(extra, node)] if extra is not None else node

    def visit_Return(self, node):
        self.generic_visit(node)
        if self.kind == "_get_best_frontier":
            if ast.unparse(node.value) != "(best_frontier, best_value)":
                raise RuntimeError("Unexpected frontier return")
            self.counts["finish"] += 1
            return [ast.copy_location(_call("finish", "self, locals()"), node), node]
        if self.kind == "_explore" and ast.unparse(node.value) == "self._stop_action":
            self.counts["stop"] += 1
            return [ast.copy_location(_call("empty_stop", "self, observations, frontiers"), node), node]
        if self.kind == "_reduce_values":
            value = ast.unparse(node.value)
            if value == "explore_values":
                self.counts["reduce_explore"] += 1
                return [ast.copy_location(_call("reduction", "self, values, 'exploration'"), node), node]
            if value == "[v[0] for v in values]":
                self.counts["reduce_target"] += 1
                return [ast.copy_location(_call("reduction", "self, values, 'target'"), node), node]
            raise RuntimeError("Unexpected V3 reduction return")
        return node


def _compile_instrumented(original, recorder, source_text):
    tree = ast.parse(textwrap.dedent(source_text))
    method = tree.body[0]
    if not isinstance(method, ast.FunctionDef) or method.decorator_list:
        raise RuntimeError("Only undecorated upstream Python methods are supported")
    transformer = _Instrument(original.__name__)
    tree = transformer.visit(copy.deepcopy(tree))
    expected = {"sort": 1, "cyclic": 1, "keep": 1, "rank": 1, "fallback": 1, "finish": 1, "stop": 0,
                "reduce_explore": 0, "reduce_target": 0}
    if original.__name__ == "_explore":
        expected = {key: (1 if key == "stop" else 0) for key in expected}
    if original.__name__ == "_reduce_values":
        expected = {key: (1 if key in ("reduce_explore", "reduce_target") else 0) for key in expected}
    if transformer.counts != expected:
        raise RuntimeError(f"Unsupported method anchors: {transformer.counts}, expected {expected}")
    if original.__name__ == "_get_best_frontier":
        # Preserve the original docstring; the observer is the first executable statement.
        body = tree.body[0].body
        offset = int(bool(body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)))
        body.insert(offset, _call("begin", "self, observations, frontiers"))
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace[_GLOBAL] = recorder
    exec(compile(tree, "<vlfm_frontier_decision_trace_20261010>", "exec"), namespace)
    return functools.update_wrapper(namespace[original.__name__], original)


class TraceHandle:
    def __init__(self, bindings):
        self.bindings = bindings
        self.installed = True

    def uninstall(self):
        if not self.installed:
            return
        for policy_class, name, original, replacement in self.bindings:
            if getattr(policy_class, name) is not replacement:
                raise RuntimeError("Refusing to overwrite a method changed by another adapter: " + name)
        for policy_class, name, original, replacement in self.bindings:
            setattr(policy_class, name, original)
        self.installed = False


def install_decision_trace(policy_class, emit, context_fn=None, reduction_class=None):
    """Install a hash-guarded observer and return a reversible TraceHandle.

    ``context_fn(policy, observations)`` should supply run_id, episode_key,
    step/action index. ``emit(record)`` normally appends strict JSONL. Trace I/O
    failures propagate: abort the diagnostic run, never silently lose evidence.
    Pass ``reduction_class=ITMPolicyV3`` to capture raw target/exploration
    channels and its actual global reduction branch. Hashes are SHA256 of
    UTF-8 LF-normalized source (Windows CRLF checkout is equivalent).
    No GPU/Habitat import is required by this file itself.
    """
    originals = [(policy_class, name, getattr(policy_class, name))
                 for name in ("_get_best_frontier", "_explore")]
    if reduction_class is not None:
        originals.append((reduction_class, "_reduce_values", getattr(reduction_class, "_reduce_values")))
    if any(hasattr(fn, "__wrapped__") for _, _, fn in originals):
        raise RuntimeError("Install before other wrappers; duplicate instrumentation is forbidden")
    for _, _, original in originals:
        filename = inspect.getsourcefile(original)
        if filename is None or hashlib.sha256(Path(filename).read_text(encoding="utf-8").encode("utf-8")).hexdigest() != EXPECTED_SOURCE_SHA256:
            raise RuntimeError("VLFM source hash mismatch: audit this revision before instrumenting")
    recorder = _Recorder(emit, context_fn)
    bindings = [(cls, name, original, _compile_instrumented(original, recorder, inspect.getsource(original)))
                for cls, name, original in originals]
    # Compile and validate both methods *before* changing either class attribute.
    for cls, name, original, replacement in bindings:
        setattr(cls, name, replacement)
    return TraceHandle(bindings)
