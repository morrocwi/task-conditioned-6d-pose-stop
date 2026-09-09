import unittest
import benchmark

class FixtureTests(unittest.TestCase):
    def test_diagnostic_controls(self):
        d = benchmark.diagnostics()
        self.assertEqual(d['task_switch']['suction'], [])
        self.assertEqual(d['task_switch']['label'], ['refine_yaw'])
        self.assertFalse(d['coupled_false_pass']['actual_coupled_success'])
        self.assertEqual(d['unreachable'], 'HOLD')
        self.assertEqual(d['no_progress_budget'], 'HOLD')
        self.assertEqual(d['greedy_trap'], {'action_greedy': 6, 'dijkstra': 5, 'idm_core': 5})

    def test_planners_match_finite_oracle(self):
        for menu in ('single', 'rich'):
            acts = benchmark.actions(menu)
            for task in benchmark.TASKS:
                for start in range(64):
                    optimum = benchmark.independent_optimum(start, task, acts)
                    for method in ('dijkstra', 'idm_core'):
                        _, _, cost, _, _ = benchmark.rollout(method, task, start, acts)
                        self.assertEqual(cost, optimum)

if __name__ == '__main__':
    unittest.main()
