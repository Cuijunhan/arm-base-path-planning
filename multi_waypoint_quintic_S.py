import numpy as np
import matplotlib.pyplot as plt

# ----------------- 归一化 7 段 S 曲线生成 -----------------
def build_normalized_s_curve(tj=0.06, tc=0.10, t4=None, n=2001):
    """
    归一化 S 曲线（τ∈[0,1]），段时长 [tj, tc, tj, t4, tj, tc, tj]
    返回：tau, s_norm, s1_norm, s2_norm, s3_norm
    """
    if t4 is None:
        t4 = 1.0 - (4.0 * tj + 2.0 * tc)
    if t4 < 0:
        raise ValueError("tj + tc 太大，导致 t4 < 0")

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
    if s_end == 0:
        raise RuntimeError("归一化积分为0")
    s_norm = s / s_end
    s1_norm = v / s_end
    s2_norm = a / s_end
    s3_norm = j_tau / s_end
    return tau, s_norm, s1_norm, s2_norm, s3_norm

# ----------------- 单段 S 曲线生成（从 q0 到 qf，给定总时间 T） -----------------
def generate_single_s_segment(q0, qf, T, tau_grid, s_norm, s1_norm, s2_norm, s3_norm, dt=0.01):
    """
    生成单段 S 曲线（位置 q, 速度 dq, 加速度 ddq, jerk dddq）在 0..T 上的采样
    使用归一化 s(τ) 和尺度变换：
      q(t)   = q0 + sign*Δ * s(τ)
      dq(t)  = sign*(Δ/T) * s1(τ)
      ddq(t) = sign*(Δ/T^2) * s2(τ)
      dddq(t)= sign*(Δ/T^3) * s3(τ)
    返回 t_local (从0开始), q,dq,ddq,dddq
    """
    if T <= 0:
        return np.array([0.0]), np.array([q0]), np.array([0.0]), np.array([0.0]), np.array([0.0])

    t_local = np.arange(0.0, T + 1e-12, dt)
    tau_samples = t_local / T
    tau_samples = np.clip(tau_samples, 0.0, 1.0)
    grid_tau = np.linspace(0.0, 1.0, len(s_norm))

    s_tau = np.interp(tau_samples, grid_tau, s_norm)
    s1_tau = np.interp(tau_samples, grid_tau, s1_norm)
    s2_tau = np.interp(tau_samples, grid_tau, s2_norm)
    s3_tau = np.interp(tau_samples, grid_tau, s3_norm)

    delta = qf - q0
    sign = np.sign(delta) if abs(delta) > 1e-12 else 1.0
    delta_abs = abs(delta)

    q = q0 + sign * delta_abs * s_tau
    dq = sign * (delta_abs / T) * s1_tau
    ddq = sign * (delta_abs / (T**2)) * s2_tau
    dddq = sign * (delta_abs / (T**3)) * s3_tau

    return t_local, q, dq, ddq, dddq

# ----------------- raised-cosine 混合窗 -----------------
def blend_weight(t, start, end, ramp_in, ramp_out):
    """
    为 segment 在全局时间 t 中计算权重（0..1）。
    ramp_in/ramp_out 为起始/结束的渐变时长（>=0）。
    在中间非渐变区权重为1；在窗外为0；在渐变区为 raised-cosine。
    """
    w = 0.0
    if t < start or t > end:
        return 0.0
    # ramp in
    if ramp_in > 1e-12 and start <= t < start + ramp_in:
        x = (t - start) / ramp_in
        return 0.5 * (1 - np.cos(np.pi * x))
    # ramp out
    if ramp_out > 1e-12 and end - ramp_out < t <= end:
        x = (t - (end - ramp_out)) / ramp_out
        # x in (0..1]
        return 0.5 * (1 + np.cos(np.pi * x))
    return 1.0

# ----------------- 多段 S 曲线拼接（含重叠融合） -----------------
def multi_segment_s_curve_blend(q_waypoints, per_segment_T, s_params, overlap_ratio=0.2, dt=0.01):
    """
    q_waypoints: list/array of waypoints (n_points, ) positions (radian) for one joint
    per_segment_T: list/array of each segment nominal time (len = n_points-1)
    s_params: (s_norm,s1,s2,s3) 归一化曲线参数
    overlap_ratio: 每段与相邻段重叠比例（相对于段时长），例如 0.2 表示重叠 20% 时间
    dt: 全局采样步长
    返回：t_global, q_global, dq_global, ddq_global, dddq_global
    """
    n_points = len(q_waypoints)
    n_segs = n_points - 1
    # 计算每段实际开始/结束时间（初始不考虑缩短）
    # start_times: 先按不重叠累计，然后我们会给每段分配 ramp_in/out
    start_times = [0.0]
    for i in range(n_segs):
        start_times.append(start_times[-1] + per_segment_T[i])
    # 现在 start_times[-1] = sum(T_i) = end_time (no blending)
    # 我们将用重叠融合：定义每段的 ramp_in 和 ramp_out
    seg_info = []
    for i in range(n_segs):
        T = per_segment_T[i]
        # overlap with previous and next
        overlap_prev = overlap_ratio * per_segment_T[i-1] if i-1 >= 0 else 0.0
        overlap_next = overlap_ratio * per_segment_T[i+1] if i+1 < n_segs else 0.0
        ramp_in = min(overlap_prev, T*0.5)
        ramp_out = min(overlap_next, T*0.5)
        # initial naive start/end (we will place segments so that start times overlap)
        seg_start = sum(per_segment_T[:i])  # naive
        seg_end = seg_start + T
        seg_info.append({
            'i': i,
            'q0': q_waypoints[i],
            'qf': q_waypoints[i+1],
            'T': T,
            'seg_start': seg_start,
            'seg_end': seg_end,
            'ramp_in': ramp_in,
            'ramp_out': ramp_out
        })

    # 为了简化：我们把每段的实际 start 调整为：seg_start' = naive_start - ramp_in
    # 使得前一段与本段在 ramp_in 时间上重叠（如果 ramp_in>0）
    for k in range(n_segs):
        seg_info[k]['actual_start'] = seg_info[k]['seg_start'] - seg_info[k]['ramp_in']
        seg_info[k]['actual_end'] = seg_info[k]['seg_end'] + seg_info[k]['ramp_out']
    # 全局起始为第一段 actual_start（应为0或小于0，裁剪为0）
    global_start = max(0.0, seg_info[0]['actual_start'])
    global_end = max(si['actual_end'] for si in seg_info)
    t_global = np.arange(global_start, global_end + 1e-12, dt)

    # 对每段生成其局部轨迹并插值到全局时间上
    s_norm, s1_norm, s2_norm, s3_norm = s_params
    seg_traj_interp = []
    for si in seg_info:
        # 局部单段在 0..T 上
        t_local, q_local, dq_local, ddq_local, dddq_local = generate_single_s_segment(
            si['q0'], si['qf'], si['T'], None, s_norm, s1_norm, s2_norm, s3_norm, dt=dt
        )
        # 把局部 t_local 平移到全局时间轴： t_abs = actual_start + t_local + ramp_in (since we shifted start earlier)
        t_abs = si['actual_start'] + t_local + si['ramp_in']
        # 插值到全局时间（超出区间交给 fill）
        q_interp = np.interp(t_global, t_abs, q_local, left=np.nan, right=np.nan)
        dq_interp = np.interp(t_global, t_abs, dq_local, left=np.nan, right=np.nan)
        ddq_interp = np.interp(t_global, t_abs, ddq_local, left=np.nan, right=np.nan)
        dddq_interp = np.interp(t_global, t_abs, dddq_local, left=np.nan, right=np.nan)
        seg_traj_interp.append({
            't_abs': t_abs,
            'q': q_interp,
            'dq': dq_interp,
            'ddq': ddq_interp,
            'dddq': dddq_interp,
            'start': si['actual_start'] + si['ramp_in'],
            'end': si['actual_end'] - si['ramp_out'],
            'ramp_in': si['ramp_in'],
            'ramp_out': si['ramp_out']
        })

    # 在每个全局时刻用 raised-cosine 窗对所有段做加权归一化混合
    q_global = np.zeros_like(t_global)
    dq_global = np.zeros_like(t_global)
    ddq_global = np.zeros_like(t_global)
    dddq_global = np.zeros_like(t_global)
    for idx_t, tg in enumerate(t_global):
        weights = []
        q_vals = []
        dq_vals = []
        ddq_vals = []
        dddq_vals = []
        for seg in seg_traj_interp:
            # 计算本段在 tg 的权重
            w = blend_weight(tg, seg['start'], seg['end'], seg['ramp_in'], seg['ramp_out'])
            # 若该段在 tg 有有效插值（非 nan），则取值
            qv = seg['q'][idx_t]
            if np.isnan(qv):
                continue
            weights.append(w)
            q_vals.append(qv)
            dq_vals.append(seg['dq'][idx_t])
            ddq_vals.append(seg['ddq'][idx_t])
            dddq_vals.append(seg['dddq'][idx_t])
        if len(weights) == 0:
            # 没段覆盖此 tg（空隙） -> keep zero or previous value
            if idx_t > 0:
                q_global[idx_t] = q_global[idx_t-1]
                dq_global[idx_t] = 0.0
                ddq_global[idx_t] = 0.0
                dddq_global[idx_t] = 0.0
            else:
                q_global[idx_t] = 0.0
            continue
        weights = np.array(weights)
        # 归一化
        sumw = np.sum(weights)
        if sumw <= 0:
            weights = np.ones_like(weights) / len(weights)
            sumw = 1.0
        weights = weights / sumw
        # 加权
        q_global[idx_t] = np.sum(weights * np.array(q_vals))
        dq_global[idx_t] = np.sum(weights * np.array(dq_vals))
        ddq_global[idx_t] = np.sum(weights * np.array(ddq_vals))
        dddq_global[idx_t] = np.sum(weights * np.array(dddq_vals))

    return t_global, q_global, dq_global, ddq_global, dddq_global

# ----------------- 多关节全体生成与同步 -----------------
def plan_multi_joint_s_curve(waypoints_deg, per_seg_Ts=None, vmax_deg_s=60.0, overlap_ratio=0.2, dt=0.01):
    """
    waypoints_deg: (n_points, n_joints) 角度矩阵（度）
    per_seg_Ts: optional list of per-segment time (len = n_points-1) — 若 None 则按距离/vmax估计
    返回：t_global, q_all, dq_all, ddq_all, dddq_all （q_all shape: n_joints x N, q in rad）
    """
    waypoints = np.deg2rad(np.asarray(waypoints_deg))
    n_points, n_joints = waypoints.shape
    n_segs = n_points - 1

    # 估计每段时间（若没有传入），以最长关节所需时间为准
    if per_seg_Ts is None:
        per_seg_Ts = []
        vmax = np.deg2rad(vmax_deg_s)
        for i in range(n_segs):
            max_needed = 0.0
            for j in range(n_joints):
                delta = abs(waypoints[i+1,j] - waypoints[i,j])
                t_est = delta / vmax if vmax > 0 else 0.1
                if t_est > max_needed:
                    max_needed = t_est
            per_seg_Ts.append(max(0.05, max_needed))  # 最小段时长 0.05s
    per_seg_Ts = np.array(per_seg_Ts, dtype=float)

    # 归一化 S 曲线参数（可以按需调 tj/tc）
    tj = 0.06
    tc = 0.10
    tau, s_norm, s1_norm, s2_norm, s3_norm = build_normalized_s_curve(tj=tj, tc=tc, t4=None, n=2001)
    s_params = (s_norm, s1_norm, s2_norm, s3_norm)

    # 对每个关节分别生成拼接轨迹（在同一全局时间网格内）
    joint_trajs = []
    t_global_ref = None
    for j in range(n_joints):
        q_way_j = waypoints[:, j]
        t_g, qg, dqg, ddqg, dddqg = multi_segment_s_curve_blend(
            q_way_j, per_seg_Ts, s_params, overlap_ratio=overlap_ratio, dt=dt
        )
        joint_trajs.append((t_g, qg, dqg, ddqg, dddqg))
        if t_global_ref is None or len(t_g) > len(t_global_ref):
            t_global_ref = t_g

    # 重新在统一 global time 上插值（使用最长网格作为全局参考）
    t_global = t_global_ref
    N = len(t_global)
    q_all = np.zeros((n_joints, N))
    dq_all = np.zeros_like(q_all)
    ddq_all = np.zeros_like(q_all)
    dddq_all = np.zeros_like(q_all)

    for j in range(n_joints):
        t_g, qg, dqg, ddqg, dddqg = joint_trajs[j]
        q_all[j] = np.interp(t_global, t_g, qg, left=qg[0], right=qg[-1])
        dq_all[j] = np.interp(t_global, t_g, dqg, left=0.0, right=0.0)
        ddq_all[j] = np.interp(t_global, t_g, ddqg, left=0.0, right=0.0)
        dddq_all[j] = np.interp(t_global, t_g, dddqg, left=0.0, right=0.0)

    return t_global, q_all, dq_all, ddq_all, dddq_all

# ----------------- 示例运行 -----------------
if __name__ == "__main__":
    # 例子：7 关节，4 个 waypoint（度制）
    # q0 = np.array([0, 10, 20, -10, 30, 0, 15], dtype=float)
    # q1 = np.array([45, 20, -10, 0, 50, -20, 0], dtype=float)
    # q2 = np.array([30, 40, 10, -20, 10, 10, 20], dtype=float)
    # q3 = np.array([60, 0, -30, 10, 0, -10, -20], dtype=float)
    q0 = np.array([0 ], dtype=float)
    q1 = np.array([30 ], dtype=float)
    q2 = np.array([45 ], dtype=float)
    q3 = np.array([60 ], dtype=float)    

    waypoints_deg = np.vstack([q0, q1, q2, q3])

    # per-segment time 可以手工给出，也可以为 None（按 vmax 自动估计）
    #per_seg_Ts = [1.5, 1.8, 2.0]   # example
    per_seg_Ts = None

    t, q_all, dq_all, ddq_all, dddq_all = plan_multi_joint_s_curve(
        waypoints_deg, per_seg_Ts=per_seg_Ts, vmax_deg_s=60.0, overlap_ratio=0.25, dt=0.01
    )

    # 绘图：位置用度显示，其他保持弧度单位
    n_joints = q_all.shape[0]
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    axs[0].set_title("Joint Position (deg)")
    for j in range(n_joints):
        axs[0].plot(t, np.rad2deg(q_all[j]), label=f"J{j+1}")
    axs[0].legend(ncol=2); axs[0].grid(True)

    axs[1].set_title("Joint Velocity (rad/s)")
    for j in range(n_joints):
        axs[1].plot(t, dq_all[j])
    axs[1].grid(True)

    axs[2].set_title("Joint Acceleration (rad/s^2)")
    for j in range(n_joints):
        axs[2].plot(t, ddq_all[j])
    axs[2].grid(True)

    axs[3].set_title("Joint Jerk (rad/s^3)")
    for j in range(n_joints):
        axs[3].plot(t, dddq_all[j])
    axs[3].grid(True)
    axs[3].set_xlabel("time (s)")
    plt.tight_layout()
    plt.show()
