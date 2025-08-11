import numpy as np
import matplotlib.pyplot as plt

def build_normalized_s_curve(tj=0.06, tc=0.10, t4=None, n=2001):
    """
    构造归一化 7 段 S 曲线（τ ∈ [0,1]）:
      段时长 = [tj, tc, tj, t4, tj, tc, tj]
      jerk_vals = [+1, 0, -1, 0, -1, 0, +1]
    返回：tau, s_norm, s1_norm, s2_norm, s3_norm
    s_norm 从 0 到 1，s1 = ds/dτ, s2 = d2s/dτ2, s3 = d3s/dτ3
    """
    # compute t4 if not given
    if t4 is None:
        t4 = 1.0 - (4.0 * tj + 2.0 * tc)
    if t4 < 0:
        raise ValueError("tj 和 tc 太大导致 t4 < 0，请减小 tj 或 tc。")

    durations = np.array([tj, tc, tj, t4, tj, tc, tj], dtype=float)
    jerk_vals = np.array([1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0], dtype=float)
    edges = np.cumsum(durations)  # 累计边界，最后应接近 1.0

    # 归一化时间网格 τ
    tau = np.linspace(0.0, 1.0, n)
    dt = tau[1] - tau[0]

    # 为每个 tau 找到对应的段索引（0..6），并裁剪防越界
    # 使用 searchsorted: 返回第一个大于 edges 的索引 => seg = idx (0..6)
    seg_idx = np.searchsorted(edges, tau, side='right')
    # seg_idx 可能为 7（当 tau==1.0 且 edges[-1]==1.0 时），需截断到 6
    seg_idx = np.minimum(seg_idx, len(jerk_vals) - 1)

    # jerk 关于 τ 的序列（piecewise 常数）
    j_tau = jerk_vals[seg_idx]

    # 用数值积分（复合梯形）从 jerk -> acc -> vel -> pos
    a = np.zeros_like(j_tau)
    v = np.zeros_like(j_tau)
    s = np.zeros_like(j_tau)
    for i in range(1, len(tau)):
        # trapezoid integrate
        a[i] = a[i-1] + 0.5 * (j_tau[i-1] + j_tau[i]) * dt
        v[i] = v[i-1] + 0.5 * (a[i-1] + a[i]) * dt
        s[i] = s[i-1] + 0.5 * (v[i-1] + v[i]) * dt

    # 归一化使 s 从 0 到 1
    s_end = s[-1]
    if s_end == 0:
        raise RuntimeError("归一化曲线积分结果为0，请检查参数。")
    s_norm = s / s_end
    s1_norm = v / s_end   # ds/dτ
    s2_norm = a / s_end   # d2s/dτ2
    s3_norm = j_tau / s_end  # d3s/dτ3 (piecewise 常数 / s_end)

    return tau, s_norm, s1_norm, s2_norm, s3_norm

def min_time_for_delta(delta_abs, vmax, amax, jmax, s1_max, s2_max, s3_max):
    """
    基于归一化曲线的最大导数，计算满足限幅的最小总时间 T
    约束：
      vmax >= (Δ * s1_max) / T   => T >= (Δ * s1_max) / vmax
      amax >= (Δ * s2_max) / T^2 => T >= sqrt( (Δ * s2_max) / amax )
      jmax >= (Δ * s3_max) / T^3 => T >= cbrt( (Δ * s3_max) / jmax )
    """
    if delta_abs <= 0:
        return 0.0
    t_v = (delta_abs * s1_max) / vmax if vmax > 0 else 0.0
    t_a = np.sqrt((delta_abs * s2_max) / amax) if amax > 0 else 0.0
    t_j = ((delta_abs * s3_max) / jmax) ** (1.0/3.0) if jmax > 0 else 0.0
    return max(t_v, t_a, t_j, 1e-8)

def generate_joint_trajectory(q0, qf, T, t_array, s_norm, s1_norm, s2_norm, s3_norm):
    """
    给定 q0,qf, 总时间 T，时间采样 t_array（秒），和归一化曲线 s(τ) 等，
    返回 q,dq,ddq,dddq 与 t_array 等长。
    注意：本实现假设初末速度/加速度为零（常见场景）。
    """
    if T <= 0:
        # 零位移或零时间，直接返回常值
        q = np.full_like(t_array, q0)
        dq = np.zeros_like(t_array)
        ddq = np.zeros_like(t_array)
        dddq = np.zeros_like(t_array)
        return q, dq, ddq, dddq

    delta = qf - q0
    sign = np.sign(delta) if abs(delta) > 0 else 1.0
    delta_abs = abs(delta)

    # τ = t / T，插值获得 s, s1, s2, s3 在对应 τ 点的值
    tau_samples = t_array / T
    tau_samples = np.clip(tau_samples, 0.0, 1.0)
    grid_tau = np.linspace(0.0, 1.0, len(s_norm))

    s_tau = np.interp(tau_samples, grid_tau, s_norm)
    s1_tau = np.interp(tau_samples, grid_tau, s1_norm)
    s2_tau = np.interp(tau_samples, grid_tau, s2_norm)
    s3_tau = np.interp(tau_samples, grid_tau, s3_norm)

    q = q0 + sign * delta_abs * s_tau
    dq = sign * (delta_abs / T) * s1_tau
    ddq = sign * (delta_abs / (T**2)) * s2_tau
    dddq = sign * (delta_abs / (T**3)) * s3_tau

    return q, dq, ddq, dddq

# -------------------- 主示例 --------------------
if __name__ == "__main__":
    # 关节数与初/末位姿（度数，内部转换为弧度）
    n_joints = 7
    q0_deg = np.array([0, 10, 20, -10, 30, 0, 15], dtype=float)
    qf_deg = np.array([45, 20, -10, 0, 50, -20, 0], dtype=float)
    # q0_deg = np.array([0], dtype=float)
    # qf_deg = np.array([90], dtype=float)
    q0 = np.deg2rad(q0_deg)
    qf = np.deg2rad(qf_deg)

    # 运动约束（可按真实机器人修改）
    vmax = 1.0    # rad/s
    amax = 5.0    # rad/s^2
    jmax = 50.0   # rad/s^3

    # 归一化 S 曲线参数（在 τ 空间）
    tj = 0.06   # 四个短坡段的时长
    tc = 0.10   # 两个中间匀加/匀减段的时长
    # t4 会自动计算为 1 - (4*tj + 2*tc)
    tau, s_norm, s1_norm, s2_norm, s3_norm = build_normalized_s_curve(tj=tj, tc=tc, t4=None, n=4001)

    # 计算归一化导数的最大值（用于时间估算）
    s1_max = np.max(np.abs(s1_norm))
    s2_max = np.max(np.abs(s2_norm))
    s3_max = np.max(np.abs(s3_norm))

    # 为每个关节估算最小时间
    min_times = []
    for i in range(n_joints):
        delta_abs = abs(qf[i] - q0[i])
        T_i = min_time_for_delta(delta_abs, vmax, amax, jmax, s1_max, s2_max, s3_max)
        min_times.append(T_i)
    min_times = np.array(min_times)

    # 同步时间取最大（保证每个关节都不超限）
    T_sync = float(np.max(min_times))
    if T_sync <= 0:
        T_sync = 0.01  # 避免 T==0 的特殊值

    print("每个关节的最小时间 (s):", np.round(min_times, 4))
    print("同步总时间 T_sync =", round(T_sync, 4), "s")

    # 时间采样（真实时间）
    dt = 0.01
    t_array = np.arange(0.0, T_sync + 1e-12, dt)

    # 生成轨迹
    q_all = np.zeros((n_joints, len(t_array)))
    dq_all = np.zeros_like(q_all)
    ddq_all = np.zeros_like(q_all)
    dddq_all = np.zeros_like(q_all)

    for i in range(n_joints):
        q, dq, ddq, dddq = generate_joint_trajectory(q0[i], qf[i], T_sync, t_array,
                                                   s_norm, s1_norm, s2_norm, s3_norm)
        q_all[i, :] = q
        dq_all[i, :] = dq
        ddq_all[i, :] = ddq
        dddq_all[i, :] = dddq

    # 绘图：位置用度显示；其余用弧度单位
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    axs[0].set_title("Joint Position (deg)")
    for i in range(n_joints):
        axs[0].plot(t_array, np.rad2deg(q_all[i]), label=f"J{i+1}")
    axs[0].legend(loc='upper right', ncol=2)
    axs[0].grid(True)

    axs[1].set_title("Joint Velocity (rad/s)")
    for i in range(n_joints):
        axs[1].plot(t_array, dq_all[i])
    axs[1].grid(True)

    axs[2].set_title("Joint Acceleration (rad/s²)")
    for i in range(n_joints):
        axs[2].plot(t_array, ddq_all[i])
    axs[2].grid(True)

    axs[3].set_title("Joint Jerk (rad/s³)")
    for i in range(n_joints):
        axs[3].plot(t_array, dddq_all[i])
    axs[3].grid(True)
    axs[3].set_xlabel("time (s)")

    plt.tight_layout()
    plt.show()
