import unittest
import numpy as np
from src.planner import SCurvePlanner

class TestSCurvePlanner(unittest.TestCase):

    def setUp(self):
        self.planner = SCurvePlanner()

    def test_build_normalized_s_curve(self):
        tau, s_norm, s1_norm, s2_norm, s3_norm = self.planner.build_normalized_s_curve(tj=0.06, tc=0.10, t4=None, n=100)
        self.assertEqual(len(tau), 100)
        self.assertAlmostEqual(s_norm[0], 0.0)
        self.assertAlmostEqual(s_norm[-1], 1.0)

    def test_min_time_for_delta(self):
        delta_abs = 1.0
        vmax = 1.0
        amax = 5.0
        jmax = 50.0
        s1_max = 1.0
        s2_max = 1.0
        s3_max = 1.0
        T = self.planner.min_time_for_delta(delta_abs, vmax, amax, jmax, s1_max, s2_max, s3_max)
        self.assertGreater(T, 0)

    def test_generate_joint_trajectory(self):
        q0 = np.array([0.0])
        qf = np.array([1.0])
        T = 1.0
        t_array = np.linspace(0, T, 100)
        tau, s_norm, s1_norm, s2_norm, s3_norm = self.planner.build_normalized_s_curve(tj=0.06, tc=0.10, t4=None, n=100)
        q, dq, ddq, dddq = self.planner.generate_joint_trajectory(q0, qf, T, t_array, s_norm, s1_norm, s2_norm, s3_norm)
        self.assertEqual(len(q), len(t_array))
        self.assertEqual(len(dq), len(t_array))
        self.assertEqual(len(ddq), len(t_array))
        self.assertEqual(len(dddq), len(t_array))

if __name__ == '__main__':
    unittest.main()