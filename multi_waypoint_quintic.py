import numpy as np
import matplotlib.pyplot as plt

# ===== quintic coeff solver =====
def quintic_coeff(q0, v0, a0, qf, vf, af, T):
    """
    返回五次多项式系数 a0..a5，使得：
    q(0)=q0, q'(0)=v0, q''(0)=a0
    q(T)=qf, q'(T)=vf, q''(T)=af
    """
    A = np.array([
        [1,    0,      0,        0,         0,      0],
        [0,    1,      0,        0,         0,      0],
        [0,    0,      2,        0,         0,      0],
        [1,    T,      T**2,     T**3,      T**4,   T**5],
        [0,    1,   2*T,     3*T**2,    4*T**3, 5*T**4],
        [0,    0,     2,     6*T,      12*T**2, 20*T**3]
    ], dtype=float)

    b = np.array([q0, v0, a0, qf, vf, af], dtype=float)
    coeffs = np.linalg.solve(A, b)
    return coeffs  # a0..a5

def eval_quintic(coeffs, t):
    a0,a1,a2,a3,a4,a5 = coeffs
    q = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    dq = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
    ddq = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
    dddq = 6*a3 + 24*a4*t + 60*a5*t**2
    return q, dq, ddq, dddq

# ===== compute via-point velocities (central difference + clamp) =====
def compute_waypoint_velocities(qs, times, vmax):
    """
    qs: (n_points,) positions (radian) for one joint
    times: (n_points,) cumulative times for waypoints (seconds)
    vmax: scalar maximum velocity for clamping
    返回 v_i for each waypoint (n_points,)
    boundary: v_0 = v_last = 0
    interior: central difference (q_{i+1}-q_{i-1})/(t_{i+1}-t_{i-1}) then clamp to vmax
    """
    n = len(qs)
    v = np.zeros(n)
    v[0] = 0.0
    v[-1] = 0.0
    for i in range(1, n-1):
        dt = times[i+1] - times[i-1]
        if dt <= 0:
            v[i] = 0.0
        else:
            vi = (qs[i+1] - qs[i-1]) / dt
            # 限幅
            if abs(vi) > vmax:
                vi = np.sign(vi) * vmax
            v[i] = vi
    return v

# ===== main multi-waypoint trajectory generator =====
def multi_waypoint_quintic(q_waypoints, waypoint_times, vmax, amax=None):
    """
    q_waypoints: (n_points, n_joints) array, 每行一个 waypoint（radian）
    waypoint_times: (n_points,) increasing cumulative time for each waypoint (seconds)
                    (如果不提供，函数可以按距离+vmax估计时间，我们这里要求传入)
    vmax: scalar or (n_joints,) maximum velocities for clamping
    返回：time_array, q_all (n_joints x N), dq_all, ddq_all, dddq_all
    """
    q_waypoints = np.asarray(q_waypoints)
    waypoint_times = np.asarray(waypoint_times)
    n_points, n_joints = q_waypoints.shape

    # 计算每个关节在每个waypoint处的速度（中心差分）
    if np.isscalar(vmax):
        vmax_vec = np.ones(n_joints) * vmax
    else:
        vmax_vec = np.asarray(vmax)

    v_waypoints = np.zeros((n_points, n_joints))
    for j in range(n_joints):
        v_waypoints[:, j] = compute_waypoint_velocities(q_waypoints[:, j], waypoint_times, vmax_vec[j])

    # 假设加速度在每个 waypoint 为 0（可扩展为估计）
    a_waypoints = np.zeros_like(v_waypoints)

    # 每段使用 quintic，从 i 到 i+1，用给定边界 pos/vel/acc，采样并拼接
    dt = 0.01
    time_list = []
    q_list = []
    dq_list = []
    ddq_list = []
    dddq_list = []

    for i in range(n_points - 1):
        T_seg = waypoint_times[i+1] - waypoint_times[i]
        if T_seg <= 0:
            continue
        t_local = np.arange(0.0, T_seg, dt)
        if i == n_points - 2:
            # 最后一段加上终点
            t_local = np.arange(0.0, T_seg + 1e-10, dt)

        # 为每个关节求系数并在 t_local 上求值
        q_seg = np.zeros((n_joints, len(t_local)))
        dq_seg = np.zeros_like(q_seg)
        ddq_seg = np.zeros_like(q_seg)
        dddq_seg = np.zeros_like(q_seg)

        for j in range(n_joints):
            coeffs = quintic_coeff(
                q_waypoints[i, j],
                v_waypoints[i, j],
                a_waypoints[i, j],
                q_waypoints[i+1, j],
                v_waypoints[i+1, j],
                a_waypoints[i+1, j],
                T_seg
            )
            for k, tt in enumerate(t_local):
                qv, dqv, ddqv, dddqv = eval_quintic(coeffs, tt)
                q_seg[j, k] = qv
                dq_seg[j, k] = dqv
                ddq_seg[j, k] = ddqv
                dddq_seg[j, k] = dddqv

        # 拼接（注意时间偏移）
        t_global = waypoint_times[i] + t_local
        time_list.append(t_global)
        q_list.append(q_seg)
        dq_list.append(dq_seg)
        ddq_list.append(ddq_seg)
        dddq_list.append(dddq_seg)

    # 把拼接片段连接成完整数组
    time_array = np.concatenate(time_list)
    q_all = np.hstack(q_list)   # shape n_joints x N
    dq_all = np.hstack(dq_list)
    ddq_all = np.hstack(ddq_list)
    dddq_all = np.hstack(dddq_list)

    return time_array, q_all, dq_all, ddq_all, dddq_all

# ================= 示例 & 绘图 =================
if __name__ == "__main__":
    # 示例：7 关节，4 个 waypoint（可以随意增减）
    # q0_deg = np.array([0, 10, 20, -10, 30, 0, 15], dtype=float)
    # q1_deg = np.array([45, 20, -10, 0, 50, -20, 0], dtype=float)
    # q2_deg = np.array([30, 40, 10, -20, 10, 10, 20], dtype=float)
    # q3_deg = np.array([60, 0, -30, 10, 0, -10, -20], dtype=float)
    q0_deg = np.array([0 ], dtype=float)
    q1_deg = np.array([45 ], dtype=float)
    q2_deg = np.array([30 ], dtype=float)
    q3_deg = np.array([60 ], dtype=float)

    q_waypoints_deg = np.vstack([q0_deg, q1_deg, q2_deg, q3_deg])
    q_waypoints = np.deg2rad(q_waypoints_deg)

    # 指定每个 waypoint 的累计时间（秒）
    # 你可以按每段距离和 vmax 估计时间，再适当修正；这里举例写死
    waypoint_times = np.array([0.0, 2.0, 4.5, 7.0])  # 必须严格递增

    vmax = np.deg2rad(60.0)  # 全关节最大速度（rad/s）

    t, q_all, dq_all, ddq_all, dddq_all = multi_waypoint_quintic(q_waypoints, waypoint_times, vmax)

    # 绘图：位置用度方便观察
    n_joints = q_waypoints.shape[1]
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axs[0].set_title("Position (deg)")
    for j in range(n_joints):
        axs[0].plot(t, np.rad2deg(q_all[j]), label=f"J{j+1}")
    axs[0].legend(ncol=2); axs[0].grid(True)

    axs[1].set_title("Velocity (rad/s)")
    for j in range(n_joints):
        axs[1].plot(t, dq_all[j])
    axs[1].grid(True)

    axs[2].set_title("Acceleration (rad/s^2)")
    for j in range(n_joints):
        axs[2].plot(t, ddq_all[j])
    axs[2].grid(True)

    axs[3].set_title("Jerk (rad/s^3)")
    for j in range(n_joints):
        axs[3].plot(t, dddq_all[j])
    axs[3].grid(True)
    axs[3].set_xlabel("time (s)")

    plt.tight_layout()
    plt.show()
