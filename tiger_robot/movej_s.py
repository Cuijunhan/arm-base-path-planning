import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# S曲线轨迹规划（简化单关节版）
# -----------------------
def plan_scurve(q0, qf, vmax, amax, jmax, dt=0.01):
    D = qf - q0
    dir = np.sign(D)
    D = abs(D)

    Tj = amax / jmax
    D_min = 2 * ((amax**2)/jmax + (amax*vmax)/jmax)

    if D < D_min:
        # 三角S曲线
        Tj = (D / (2*jmax))**(1/3)
        Ta = 2*Tj
        Tv = 0
    else:
        Ta = amax/jmax + (vmax/amax)
        Tv = (D - 2*(amax**2)/jmax) / vmax

    Td = Ta
    T = Ta + Tv + Td

    times = np.arange(0, T+dt, dt)
    q, qd, qdd = [], [], []

    for t in times:
        if t < Tj:
            a = jmax * t
            v = 0.5*jmax*t**2
            p = (jmax/6)*t**3
        elif t < Ta - Tj:
            a = amax
            v = 0.5*jmax*Tj**2 + amax*(t - Tj)
            p = (jmax/6)*Tj**3 + 0.5*amax*(t - Tj)**2 + amax*Tj*(t - Tj)
        elif t < Ta:
            dt2 = t - (Ta - Tj)
            a = amax - jmax*dt2
            v = 0.5*jmax*Tj**2 + amax*(Ta - 2*Tj) + amax*dt2 - 0.5*jmax*dt2**2
            p = (jmax/6)*Tj**3 + 0.5*amax*(Ta - 2*Tj)**2 + amax*Tj*(Ta - 2*Tj) + amax*dt2 - (jmax/6)*dt2**3
        elif t < Ta + Tv:
            a = 0
            v = vmax
            p = 0.5*amax*Ta**2 + vmax*(t - Ta)
        elif t < Ta + Tv + Tj:
            dt2 = t - (Ta + Tv)
            a = -jmax*dt2
            v = vmax - 0.5*jmax*dt2**2
            p = 0.5*amax*Ta**2 + vmax*Tv + vmax*dt2 - (jmax/6)*dt2**3
        elif t < Ta + Tv + Td - Tj:
            a = -amax
            v = vmax - amax*(t - (Ta + Tv + Tj))
            p = 0.5*amax*Ta**2 + vmax*Tv + vmax*(t - (Ta + Tv)) - 0.5*amax*(t - (Ta + Tv + Tj))**2
        else:
            dt2 = t - (Ta + Tv + Td - Tj)
            a = -amax + jmax*dt2
            v = vmax - amax*(Td - Tj) + amax*dt2 - 0.5*jmax*dt2**2
            p = D
        q.append(q0 + dir*p)
        qd.append(dir*v)
        qdd.append(dir*a)

    return times, np.array(q), np.array(qd), np.array(qdd)

# 示例参数
q0 = 0
qf = 1.0
vmax = 1.0
amax = 2.0
jmax = 10.0
dt = 0.01

t, q, qd, qdd = plan_scurve(q0, qf, vmax, amax, jmax, dt)

# -----------------------
# 绘图
# -----------------------
plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.plot(t, q)
plt.ylabel("关节角度 (rad)")

plt.subplot(3,1,2)
plt.plot(t, qd)
plt.ylabel("关节速度 (rad/s)")

plt.subplot(3,1,3)
plt.plot(t, qdd)
plt.ylabel("关节加速度 (rad/s²)")
plt.xlabel("时间 (s)")
plt.tight_layout()
plt.show()
