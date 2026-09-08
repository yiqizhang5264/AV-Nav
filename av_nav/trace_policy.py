from habitat_baselines.common.baseline_registry import baseline_registry
from vlfm.policy.habitat_policies import HabitatITMPolicyV2
from .trace import TraceMixin


@baseline_registry.register_policy
class TracedVLFMPolicy(TraceMixin,HabitatITMPolicyV2):
    """Original pinned VLFM actions with passive instrumentation only."""
