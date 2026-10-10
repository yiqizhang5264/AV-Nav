"""Install once, only from the dedicated frontier-probe runner."""
_HANDLE = None


def install_policy_hooks():
    global _HANDLE
    if _HANDLE is not None:
        return
    from vlfm.policy.itm_policy import BaseITMPolicy, ITMPolicyV3
    from decision_trace import install_decision_trace
    from frontier_probe_recorder import get_probe
    _HANDLE = install_decision_trace(BaseITMPolicy, get_probe().decision,
                                     reduction_class=ITMPolicyV3)
