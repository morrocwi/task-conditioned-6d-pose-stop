import importlib.util
import inspect
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("labrun", ROOT/"lab"/"run_real_system.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class TestLabHarness(unittest.TestCase):
    def test_gate_has_no_oracle_argument(self):
        params=set(inspect.signature(m.first_certificate_stage).parameters)
        self.assertEqual(params,{"stages","spec","model","q"})

    def test_split_leakage_rejected(self):
        row={"episode_id":"x","stages":[{"k":0,"features":[1.0],"incremental_ms":1.0,"estimator_stop":True}],"oracle":{"abs_pose_error_6d":[[0,0,0,0,0,0]]}}
        with self.assertRaises(ValueError):
            m.ensure_disjoint(("train",[row]),("test",[row]))

    def test_conformal_quantile(self):
        q,rank=m.conformal_quantile([1,2,3,4,5,6,7,8,9,10],0.2)
        self.assertEqual(rank,9); self.assertEqual(q,9)

    def test_robust_l1_is_joint(self):
        spec={"kind":"l1","mask":[0,1,5],"tol":[1,1,99,99,99,1]}
        self.assertFalse(m.task_pass(spec,[0.4,0.4,0,0,0,0.4]))
        self.assertTrue(m.task_pass(spec,[0.2,0.2,0,0,0,0.2]))

if __name__=="__main__": unittest.main()
