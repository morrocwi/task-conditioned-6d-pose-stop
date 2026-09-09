import importlib.util, math, sys, unittest
from pathlib import Path

P=Path(__file__).resolve().parents[1]/'theory'/'completion_envelope.py'
spec=importlib.util.spec_from_file_location('completion_envelope',P)
m=importlib.util.module_from_spec(spec); sys.modules['completion_envelope']=m; spec.loader.exec_module(m)

class TestCompletionEnvelope(unittest.TestCase):
    def test_task_switch_same_unknown_yaw(self):
        b=(.002,.002,.004,math.radians(2),math.radians(2),math.radians(12))
        self.assertEqual(m.envelope_verdict('top_suction',b),m.PASS)
        self.assertEqual(m.envelope_verdict('label_alignment',b),m.HOLD)
    def test_refining_only_yaw_resolves_label(self):
        b=(.002,.002,.004,math.radians(2),math.radians(2),math.radians(12))
        a=(*b[:5],math.radians(2))
        self.assertEqual(m.contraction_value('label_alignment',b,a),1)
        self.assertEqual(m.envelope_verdict('label_alignment',a),m.PASS)
    def test_coupled_insertion_detects_joint_ambiguity(self):
        b=(.003,.003,0.0,0.0,0.0,math.radians(2.4))
        # Each active marginal bound is below its own declared tolerance,
        # but the L1 task reader can still fail at joint corners.
        self.assertLess(b[0],.004); self.assertLess(b[1],.004); self.assertLess(b[5],math.radians(3))
        self.assertEqual(m.envelope_verdict('keyed_insertion',b),m.HOLD)
    def test_finite_pass_is_reader_specific_not_pose_identity(self):
        b=(.001,.001,.002,math.radians(1),math.radians(1),math.radians(20))
        self.assertEqual(m.envelope_verdict('top_suction',b),m.PASS)
        # Pose remains non-singleton: there are multiple completions.
        self.assertGreater(len(m.box_completion_set(b)),1)
    def test_action_ranking_uses_declared_model_only(self):
        b=(.002,.002,.004,math.radians(2),math.radians(2),math.radians(12))
        yaw=(*b[:5],math.radians(2)); nohelp=(.001,.001,.003,math.radians(1),math.radians(1),math.radians(12))
        actions=[('refine_yaw',yaw,2),('refine_translation',nohelp,1)]
        self.assertEqual(m.choose_by_contraction_per_cost('label_alignment',b,actions),'refine_yaw')

if __name__=='__main__': unittest.main()
