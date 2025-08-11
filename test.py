import numpy as np
import matplotlib.pyplot as plt

def quintic_coeff(q0, v0, a0, qf, vf, af, T):
    a0_c = q0
    a1_c = v0
    a2_c = a0 / 2
    a3_c = (20*(qf - q0) - (8*vf + 12*v0)*T - (3*a0 - af)*T**2) / (2*T**3)
    a4_c = (30*(q0 - qf) + (14*vf + 16*v0)*T + (3*a0 - 2*af)*T**2) / (2*T**4)
    a5_c = (12*(qf - q0) - (6*vf + 6*v0)*T - (a0 - af)*T**2) / (2*T**5)
    return [a0_c, a1_c, a2_c, a3_c, a4_c, a5_c]

def quintic_trajectory(a, t):
    q = a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4 + a[5]*t**5
    dq = a[1] + 2*a[2]*t + 3*a[3]*t**2 + 4*a[4]*t**3 + 5*a[5]*t**4
    ddq = 2*a[2] + 6*a[3]*t + 12*a[4]*t**2 + 20*a[5]*t**3
    dddq = 6*a[3] + 24*a[4]*t + 60*a[5]*t**2
    return q, dq, ddq, dddq

# ===== 输入参数 =====
n_joints = 7
q0_list = np.deg2rad([0, 10, 20, -10, 30, 0, 15])   # 起点角度
qf_list = np.deg2rad([45, 20, -10, 0, 50, -20, 0])  # 终点角度
v0_list = np.zeros(n_joints)
vf_list = np.zeros(n_joints)
a0_list = np.zeros(n_joints)
af_list = np.zeros(n_joints)

# 单关节时间估计（简单按最大速度 30°/s 估算）
max_vel = np.deg2rad(30)
times = np.abs(qf_list - q0_list) / max_vel
T_sync = np.max(times)  # 同步时间

# ===== 轨迹计算 =====
dt = 0.01
t_array = np.arange(0, T_sync + dt, dt)

q_all, dq_all, ddq_all, dddq_all = [], [], [], []

for i in range(n_joints):
    coeffs = quintic_coeff(q0_list[i], v0_list[i], a0_list[i],
                           qf_list[i], vf_list[i], af_list[i],
                           T_sync)
    q_traj, dq_traj, ddq_traj, dddq_traj = [], [], [], []
    for t in t_array:
        q, dq, ddq, dddq = quintic_trajectory(coeffs, t)
        q_traj.append(q)
        dq_traj.append(dq)
        ddq_traj.append(ddq)
        dddq_traj.append(dddq)
    q_all.append(q_traj)
    dq_all.append(dq_traj)
    ddq_all.append(ddq_traj)
    dddq_all.append(dddq_traj)

q_all = np.array(q_all)
dq_all = np.array(dq_all)
ddq_all = np.array(ddq_all)
dddq_all = np.array(dddq_all)

# ===== 绘图 =====
fig, axs = plt.subplots(4, 1, figsize=(10, 12))
titles = ["Joint Position (rad)", "Joint Velocity (rad/s)",
          "Joint Acceleration (rad/s²)", "Joint Jerk (rad/s³)"]
data_all = [q_all, dq_all, ddq_all, dddq_all]

for idx, ax in enumerate(axs):
    for j in range(n_joints):
        ax.plot(t_array, data_all[idx][j], label=f"Joint {j+1}")
    ax.set_title(titles[idx])
    ax.grid(True)
    if idx == 0:
        ax.legend()
plt.tight_layout()
plt.show()
