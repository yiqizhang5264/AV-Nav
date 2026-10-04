import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('sap_compare',Path(__file__).resolve().parents[1]/'scripts/compare_sap_suites.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SAPComparisonTests(unittest.TestCase):
    def records(self):
        return {str(i):dict(scene_id='scene',episode_id='0',episode_sha256=str(i),source_file='scene.gz',
            source_row=i,metrics=dict(success=s,spl=s*.5,target_object='bed',sap_triggers=0,
                                     sap_vlm_calls=2,sap_repositions=0)) for i,s in enumerate([0,1])}

    def test_repeated_original_ids_remain_two_pairs(self):
        base=self.records()
        sap=copy.deepcopy(base)
        sap['0']['metrics'].update(success=1,spl=.8,sap_triggers=1,sap_repositions=2)
        sap['1']['metrics'].update(success=0,spl=0)
        result=module.pair_records(base,sap)['overall']
        self.assertEqual(result['episodes'],2)
        self.assertEqual(result['recovered'],1)
        self.assertEqual(result['regressed'],1)
        self.assertEqual(result['zero_trigger_episodes'],1)
        self.assertAlmostEqual(result['delta']['spl'],.15)

    def test_mismatched_episode_payload_rejected(self):
        base=self.records()
        sap=copy.deepcopy(base)
        sap['0']['episode_sha256']='different'
        with self.assertRaisesRegex(ValueError,'metadata'):
            module.pair_records(base,sap)

    def test_partial_pairing_rejected(self):
        base=self.records()
        with self.assertRaisesRegex(ValueError,'source-row'):
            module.pair_records(base,{'0':base['0']})
