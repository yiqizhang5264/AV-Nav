import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('select_ocs_cases', Path(__file__).parents[1]/'scripts/select_ocs_cases.py')
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class OCSSelectionTests(unittest.TestCase):
    def test_hash_selection_independent_of_input_order(self):
        groups = {s: [dict(dataset='hm3dv1',scene='abc',episode_key=f'abc.basis.glb_{i}',ocs_stratum=s)
                      for i in range(20)] for s in mod.STRATA}
        a, _ = mod.select(groups, 4)
        b, _ = mod.select({s:list(reversed(rows)) for s,rows in groups.items()},4)
        self.assertEqual(a,b)
        self.assertEqual([r['ocs_stratum'] for r in a[:3]], list(mod.STRATA))
        self.assertEqual(len(a),12)

    def test_shortage_not_silently_relaxed(self):
        with self.assertRaises(ValueError): mod.select({s:[] for s in mod.STRATA},1)

    def test_missing_audit_is_not_negative_control(self):
        rows=[dict(dataset='hm3dv1',episode_key='abc.basis.glb_1',strict_2d_input_eligible=True)]
        out=mod.candidates({'episodes':rows},{'cases':[]},{})
        self.assertEqual(sum(map(len,out.values())),0)

    def test_unknown_audit_not_negative_control(self):
        row=dict(dataset='hm3dv1',episode_key='abc.basis.glb_1',strict_2d_input_eligible=True)
        audit=dict(row,status='unknown',episode_category='no_qualifying_spatial_return')
        out=mod.candidates({'episodes':[row]},{'cases':[audit]}, {})
        self.assertEqual(sum(map(len,out.values())),0)

    def test_true_failure_strata(self):
        row=dict(dataset='hm3dv1',episode_key='abc.basis.glb_1',strict_2d_input_eligible=True)
        audit=dict(row,status='analyzed',episode_category='contains_low_yield_repeated_return')
        out=mod.candidates({'episodes':[row]},{'cases':[audit]}, {})
        self.assertEqual(len(out[mod.STRATA[0]]),1)
        self.assertEqual(len(out[mod.STRATA[1]]),0)


if __name__ == '__main__': unittest.main()
