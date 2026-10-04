"""Load the explicit SAP adapter, then run the pinned Habitat evaluator."""
import os
import runpy
from av_nav.runtime import install

install()
if os.environ['SAP_VARIANT'] != 'baseline':
    import av_nav.sap_policy  # noqa: F401
runpy.run_module('vlfm.run', run_name='__main__')
