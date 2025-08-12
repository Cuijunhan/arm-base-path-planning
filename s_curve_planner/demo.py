import numpy as np
from s_curve_planner import SCurvePlanner
from s_curve_visualizer import SCurveVisualizer

# 关节参数
n_joints = 7
q0_deg = np.array([0, 10, 20, -10, 30, 0, 15], dtype=float)
qf_deg = np.array([45, 20, -10, 0, 50, -20, 0], dtype=float)
q0 = np.deg2rad(q0_deg)
qf = np.deg2rad(qf_deg)

vmax = 1.0
amax = 5.0
jmax = 50.0

planner = SCurvePlanner()
s1_max = np.max(np.abs(planner.s1_norm))
s2_max = np.max(np.abs(planner.s2_norm))
s3_max = np.max(np.abs(planner.s3_norm))

min_times = []
for i in range(n_joints):
    delta_abs = abs(qf[i] - q0[i])
    T_i = planner.min_time_for_delta(delta_abs, vmax, amax, jmax, s1_max, s2_max, s3_max)
    min_times.append(T_i)
T_sync = float(np.max(min_times))
dt = 0.01
t_array = np.arange(0.0, T_sync + 1e-12, dt)

q_all = np.zeros((n_joints, len(t_array)))
dq_all = np.zeros_like(q_all)
ddq_all = np.zeros_like(q_all)
dddq_all = np.zeros_like(q_all)
for i in range(n_joints):
    q, dq, ddq, dddq = planner.generate_joint_trajectory(q0[i], qf[i], T_sync, t_array)
    q_all[i, :] = q
    dq_all[i, :] = dq
    ddq_all[i, :] = ddq
    dddq_all[i, :] = dddq

SCurveVisualizer.plot_trajectory(t_array, q_all, dq_all, ddq_all, dddq_all)