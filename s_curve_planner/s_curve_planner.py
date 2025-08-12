import numpy as np

class SCurvePlanner:
    def __init__(self, tj=0.06, tc=0.10, n=2001):
        self.tj = tj
        self.tc = tc
        self.n = n
        self.tau, self.s_norm, self.s1_norm, self.s2_norm, self.s3_norm = self.build_normalized_s_curve()

    def build_normalized_s_curve(self, tj=None, tc=None, t4=None, n=None):
        tj = tj if tj is not None else self.tj
        tc = tc if tc is not None else self.tc
        n = n if n is not None else self.n
        if t4 is None:
            t4 = 1.0 - (4.0 * tj + 2.0 * tc)
        durations = np.array([tj, tc, tj, t4, tj, tc, tj], dtype=float)
        jerk_vals = np.array([1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0], dtype=float)
        edges = np.cumsum(durations)
        tau = np.linspace(0.0, 1.0, n)
        dt = tau[1] - tau[0]
        seg_idx = np.searchsorted(edges, tau, side='right')
        seg_idx = np.minimum(seg_idx, len(jerk_vals) - 1)
        j_tau = jerk_vals[seg_idx]
        a = np.zeros_like(j_tau)
        v = np.zeros_like(j_tau)
        s = np.zeros_like(j_tau)
        for i in range(1, len(tau)):
            a[i] = a[i-1] + 0.5 * (j_tau[i-1] + j_tau[i]) * dt
            v[i] = v[i-1] + 0.5 * (a[i-1] + a[i]) * dt
            s[i] = s[i-1] + 0.5 * (v[i-1] + v[i]) * dt
        s_end = s[-1]
        s_norm = s / s_end
        s1_norm = v / s_end
        s2_norm = a / s_end
        s3_norm = j_tau / s_end
        return tau, s_norm, s1_norm, s2_norm, s3_norm

    def min_time_for_delta(self, delta_abs, vmax, amax, jmax, s1_max, s2_max, s3_max):
        if delta_abs <= 0:
            return 0.0
        t_v = (delta_abs * s1_max) / vmax if vmax > 0 else 0.0
        t_a = np.sqrt((delta_abs * s2_max) / amax) if amax > 0 else 0.0
        t_j = ((delta_abs * s3_max) / jmax) ** (1.0/3.0) if jmax > 0 else 0.0
        return max(t_v, t_a, t_j, 1e-8)

    def generate_joint_trajectory(self, q0, qf, T, t_array):
        delta = qf - q0
        sign = np.sign(delta) if abs(delta) > 0 else 1.0
        delta_abs = abs(delta)
        tau_samples = t_array / T
        tau_samples = np.clip(tau_samples, 0.0, 1.0)
        grid_tau = np.linspace(0.0, 1.0, len(self.s_norm))
        s_tau = np.interp(tau_samples, grid_tau, self.s_norm)
        s1_tau = np.interp(tau_samples, grid_tau, self.s1_norm)
        s2_tau = np.interp(tau_samples, grid_tau, self.s2_norm)
        s3_tau = np.interp(tau_samples, grid_tau, self.s3_norm)
        q = q0 + sign * delta_abs * s_tau
        dq = sign * (delta_abs / T) * s1_tau
        ddq = sign * (delta_abs / (T**2)) * s2_tau
        dddq = sign * (delta_abs / (T**3)) * s3_tau
        return q, dq, ddq, dddq