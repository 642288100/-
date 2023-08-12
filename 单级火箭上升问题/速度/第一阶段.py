import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

# 参数设置
m0 = 1200  # 初始质量（kg）
u = 15     # 燃料消耗速率（kg/s）
F = 27000   # 推力（N）
k = 0.4    # 空气阻力比例系数（kg/m）
g = 9.8    # 重力加速度（m/s^2）
t1 = 1100 / u  # 引擎关闭时间（s）

# 第一阶段速度函数
def velocity_prime_stage1(t, v):
    denominator = m0 - u * t
    if denominator == 0:
        print("分母为0")  # 或者返回其他合适的值，避免除以零
    return (F - k * v**2) / float(denominator) - g

# 使用经典四阶Runge-Kutta方法求解速度
def runge_kutta_4(f, t0, v0, dt, steps):
    t_values = [t0]
    v_values = [v0]
    for _ in range(steps):
        k1 = f(t0, v0)
        k2 = f(t0 + dt / 2, v0 + dt / 2 * k1)
        k3 = f(t0 + dt / 2, v0 + dt / 2 * k2)
        k4 = f(t0 + dt, v0 + dt * k3)
        
        v0 = v0 + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        t0 = t0 + dt
        
        t_values.append(t0)
        v_values.append(v0)
    return t_values, v_values

h = 20000000 # 步长

# 求解第一阶段的速度
t_values, v_values = runge_kutta_4(velocity_prime_stage1, 0, 0, t1 / h, h)  # 使用1000倍的步数来更精确地近似

# 输出第一阶段结束时的速度
print("第一阶段结束时的速度：", v_values[-1])

'''
# 绘图
plt.plot(t_values, v_values)
plt.xlabel('time (s)')
plt.ylabel('speed (m/s)')
plt.title('Rocket First Stage Velocity Variation(Runge-Kutta method)')
plt.grid(True)
plt.show()

#保存绘图
filename = "velocity_stage1.png"
plt.savefig(filename)
plt.close()
'''


