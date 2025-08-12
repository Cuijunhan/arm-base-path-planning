import numpy as np

class SCurvePlanner:
    """
    S曲线轨迹规划器
    
    该类实现了基于S曲线的机器人关节轨迹规划，通过控制加加速度(jerk)来生成平滑的轨迹。
    S曲线由7个阶段组成：加加速、匀速加速、减加速、匀速、加减速、匀速减速、减减速。
    
    主要功能：
    1. 构建标准化的S曲线模板
    2. 计算满足约束条件的最小运动时间
    3. 生成完整的关节轨迹（位置、速度、加速度、加加速度）
    """
    
    def __init__(self, tj=0.06, tc=0.10, n=2001):
        """
        初始化S曲线规划器
        
        参数:
            tj (float): 加加速时间，默认0.06秒（对应S曲线的第1、3、5、7阶段）
            tc (float): 匀速时间，默认0.10秒（对应S曲线的第2、6阶段）
            n (int): 轨迹采样点数，默认2001个点
        """
        self.tj = tj  # 加加速时间
        self.tc = tc  # 匀速时间
        self.n = n    # 采样点数
        
        # 构建标准化的S曲线模板
        # tau: 归一化时间 [0,1]
        # s_norm: 归一化位置曲线
        # s1_norm: 归一化速度曲线
        # s2_norm: 归一化加速度曲线
        # s3_norm: 归一化加加速度曲线
        self.tau, self.s_norm, self.s1_norm, self.s2_norm, self.s3_norm = self.build_normalized_s_curve()

    def build_normalized_s_curve(self, tj=None, tc=None, t4=None, n=None):
        """
        构建标准化的S曲线模板
        
        S曲线分为7个阶段：
        阶段1: 加加速 (jerk = +1)
        阶段2: 匀速加速 (jerk = 0)
        阶段3: 减加速 (jerk = -1)
        阶段4: 匀速 (jerk = 0)
        阶段5: 加减速 (jerk = -1)
        阶段6: 匀速减速 (jerk = 0)
        阶段7: 减减速 (jerk = +1)
        
        参数:
            tj (float): 加加速时间
            tc (float): 匀速时间
            t4 (float): 中间匀速段时间
            n (int): 采样点数
            
        返回:
            tuple: (tau, s_norm, s1_norm, s2_norm, s3_norm)
                - tau: 归一化时间数组 [0,1]
                - s_norm: 归一化位置曲线
                - s1_norm: 归一化速度曲线
                - s2_norm: 归一化加速度曲线
                - s3_norm: 归一化加加速度曲线
        """
        # 使用默认参数或传入的参数
        tj = tj if tj is not None else self.tj
        tc = tc if tc is not None else self.tc
        n = n if n is not None else self.n
        
        # 计算中间匀速段时间，确保总时间为1
        if t4 is None:
            t4 = 1.0 - (4.0 * tj + 2.0 * tc)
        
        # 定义7个阶段的时间长度
        durations = np.array([tj, tc, tj, t4, tj, tc, tj], dtype=float)
        
        # 定义7个阶段的加加速度值（归一化）
        jerk_vals = np.array([1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0], dtype=float)
        
        # 计算各阶段的累积时间边界
        edges = np.cumsum(durations)
        
        # 生成归一化时间数组
        tau = np.linspace(0.0, 1.0, n)
        dt = tau[1] - tau[0]  # 时间步长
        
        # 确定每个时间点属于哪个阶段
        seg_idx = np.searchsorted(edges, tau, side='right')
        seg_idx = np.minimum(seg_idx, len(jerk_vals) - 1)
        
        # 根据阶段索引获取对应的加加速度值
        j_tau = jerk_vals[seg_idx]
        
        # 初始化位置、速度、加速度数组
        a = np.zeros_like(j_tau)  # 加速度
        v = np.zeros_like(j_tau)  # 速度
        s = np.zeros_like(j_tau)  # 位置
        
        # 通过数值积分计算位置、速度、加速度
        # 使用梯形积分方法
        for i in range(1, len(tau)):
            # 积分加加速度得到加速度
            a[i] = a[i-1] + 0.5 * (j_tau[i-1] + j_tau[i]) * dt
            # 积分加速度得到速度
            v[i] = v[i-1] + 0.5 * (a[i-1] + a[i]) * dt
            # 积分速度得到位置
            s[i] = s[i-1] + 0.5 * (v[i-1] + v[i]) * dt
        
        # 获取最终位置值用于归一化
        s_end = s[-1]
        
        # 归一化所有曲线，使最终位置为1
        s_norm = s / s_end      # 归一化位置曲线
        s1_norm = v / s_end     # 归一化速度曲线
        s2_norm = a / s_end     # 归一化加速度曲线
        s3_norm = j_tau / s_end # 归一化加加速度曲线
        
        return tau, s_norm, s1_norm, s2_norm, s3_norm

    def min_time_for_delta(self, delta_abs, vmax, amax, jmax, s1_max, s2_max, s3_max):
        """
        计算满足约束条件的最小运动时间
        
        根据速度、加速度、加加速度约束，计算完成指定位移所需的最小时间。
        取三个约束中的最大值作为最终时间。
        
        参数:
            delta_abs (float): 绝对位移距离
            vmax (float): 最大速度约束
            amax (float): 最大加速度约束
            jmax (float): 最大加加速度约束
            s1_max (float): 归一化速度曲线的最大值
            s2_max (float): 归一化加速度曲线的最大值
            s3_max (float): 归一化加加速度曲线的最大值
            
        返回:
            float: 满足所有约束的最小运动时间
        """
        # 如果位移为0，返回0时间
        if delta_abs <= 0:
            return 0.0
        
        # 根据速度约束计算最小时间
        # T = delta * s1_max / vmax
        t_v = (delta_abs * s1_max) / vmax if vmax > 0 else 0.0
        
        # 根据加速度约束计算最小时间
        # T = sqrt(delta * s2_max / amax)
        t_a = np.sqrt((delta_abs * s2_max) / amax) if amax > 0 else 0.0
        
        # 根据加加速度约束计算最小时间
        # T = (delta * s3_max / jmax)^(1/3)
        t_j = ((delta_abs * s3_max) / jmax) ** (1.0/3.0) if jmax > 0 else 0.0
        
        # 返回三个约束中的最大值，确保满足所有约束
        return max(t_v, t_a, t_j, 1e-8)

    def generate_joint_trajectory(self, q0, qf, T, t_array):
        """
        生成关节轨迹
        
        根据起始位置、目标位置、运动时间和时间数组，生成完整的S曲线轨迹，
        包括位置、速度、加速度和加加速度。
        
        参数:
            q0 (float): 起始关节位置
            qf (float): 目标关节位置
            T (float): 总运动时间
            t_array (array): 时间采样数组，包含轨迹的所有时间点
            
        返回:
            tuple: (q, dq, ddq, dddq)
                - q: 位置轨迹数组
                - dq: 速度轨迹数组
                - ddq: 加速度轨迹数组
                - dddq: 加加速度轨迹数组
        """
        # 计算位置差值和运动方向
        delta = qf - q0  # 位置差值
        sign = np.sign(delta) if abs(delta) > 0 else 1.0  # 运动方向符号，避免除零错误
        delta_abs = abs(delta)  # 绝对位置差值
        
        # 将实际时间转换为归一化时间 [0,1]
        tau_samples = t_array / T  # 归一化时间数组
        tau_samples = np.clip(tau_samples, 0.0, 1.0)  # 限制在[0,1]范围内，防止超出边界
        
        # 创建归一化S曲线模板的时间网格
        grid_tau = np.linspace(0.0, 1.0, len(self.s_norm))  # 模板的归一化时间网格
        
        # 通过插值获取对应时间点的归一化轨迹值
        s_tau = np.interp(tau_samples, grid_tau, self.s_norm)    # 插值得到归一化位置
        s1_tau = np.interp(tau_samples, grid_tau, self.s1_norm)  # 插值得到归一化速度
        s2_tau = np.interp(tau_samples, grid_tau, self.s2_norm)  # 插值得到归一化加速度
        s3_tau = np.interp(tau_samples, grid_tau, self.s3_norm)  # 插值得到归一化加加速度
        
        # 根据S曲线模板和实际参数计算最终的轨迹
        # 位置轨迹：q = q0 + sign * delta_abs * s_tau
        q = q0 + sign * delta_abs * s_tau
        
        # 速度轨迹：dq = sign * (delta_abs / T) * s1_tau
        # 速度缩放因子：delta_abs / T
        dq = sign * (delta_abs / T) * s1_tau
        
        # 加速度轨迹：ddq = sign * (delta_abs / T^2) * s2_tau
        # 加速度缩放因子：delta_abs / T^2
        ddq = sign * (delta_abs / (T**2)) * s2_tau
        
        # 加加速度轨迹：dddq = sign * (delta_abs / T^3) * s3_tau
        # 加加速度缩放因子：delta_abs / T^3
        dddq = sign * (delta_abs / (T**3)) * s3_tau
        
        return q, dq, ddq, dddq