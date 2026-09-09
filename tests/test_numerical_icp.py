import importlib.util, math, sys, tempfile, unittest
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parents[1]/'experiments'/'numerical_icp.py'
spec=importlib.util.spec_from_file_location('numerical_icp',P)
m=importlib.util.module_from_spec(spec); sys.modules['numerical_icp']=m; spec.loader.exec_module(m)

class TestNumericalICP(unittest.TestCase):
    def test_kabsch_exact_correspondences(self):
        rng=np.random.default_rng(7); A=rng.normal(size=(40,3)); R=m.rodrigues(np.array([.2,-.1,.15])); t=np.array([.03,-.02,.4]); B=m.apply(R,t,A)
        Re,te=m.best_fit(A,B)
        self.assertTrue(np.allclose(Re,R,atol=1e-10)); self.assertTrue(np.allclose(te,t,atol=1e-10))
    def test_pose_error_identity(self):
        R=m.rodrigues(np.array([.1,.2,-.1])); t=np.array([.1,-.2,.3]); self.assertLess(float(np.linalg.norm(m.pose_error(R,t,R,t))),1e-9)
    def test_yaw_irrelevant_only_for_suction(self):
        e=np.zeros(6); e[5]=math.radians(20)
        self.assertTrue(m.task_success('top_suction',e)); self.assertFalse(m.task_success('label_alignment',e))
    def test_coupled_insertion_rejects_marginal_pass(self):
        e=np.zeros(6); e[0]=.003; e[5]=math.radians(2.4)
        self.assertLess(abs(e[0]),.004); self.assertLess(abs(e[5]),math.radians(3)); self.assertFalse(m.task_success('keyed_insertion',e))
    def test_gate_has_no_ground_truth_argument(self):
        import inspect
        self.assertEqual(str(inspect.signature(m.observable_task_score)),'(task, stage)')
    def test_split_seeds_disjoint(self):
        self.assertNotIn(m.TUNE_SEED,m.TEST_SEEDS); self.assertNotIn(m.SAFETY_SEED,m.TEST_SEEDS); self.assertNotEqual(m.TUNE_SEED,m.SAFETY_SEED)
    def test_ci_run_schema(self):
        with tempfile.TemporaryDirectory() as d:
            p=m.run('ci',d)
            self.assertEqual(p['label'],'[NumericalBackend]')
            b=p['evidence_boundary']
            self.assertTrue(b['rigid_registration_executed']); self.assertFalse(b['gate_has_hidden_ground_truth_access'])
            self.assertFalse(b['camera_executed']); self.assertFalse(b['neural_pose_model_executed']); self.assertFalse(b['physical_robot_executed'])

if __name__=='__main__': unittest.main()
