import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

# 参数设置
m0 = 1200  # 初始质量（kg）
u = 15     # 燃料消耗速率（kg/s）
F = 27000   # 推力（N）
k = 0.4    # 空气阻力比例系数（kg/m）
g = 9.8    # 重力加速度（m/s^2）
t1 = m0 / u  # 引擎关闭时间（s）

# 第一阶段速度函数
def velocity_prime_stage1(t, v):
    denominator = m0 - u * t
    if denominator == 0:
        print("分母为0")  # 或者返回其他合适的值，避免除以零
    return (F - k * v**2) / float(denominator) - g

# 使用经典四阶Runge-Kutta方法求解速度和加速度
def runge_kutta_4(f, t0, v0, a0, dt, steps):
    t_values = [t0]
    v_values = [v0]
    a_values = [a0]
    for _ in range(steps):
        k1_v = f(t0, v0)
        k1_a = (F - k * v0**2) / float(m0 - u * t0) - g
        
        k2_v = f(t0 + dt / 2, v0 + dt / 2 * k1_v)
        k2_a = (F - k * (v0 + dt / 2 * k1_v)**2) / float(m0 - u * (t0 + dt / 2)) - g
        
        k3_v = f(t0 + dt / 2, v0 + dt / 2 * k2_v)
        k3_a = (F - k * (v0 + dt / 2 * k2_v)**2) / float(m0 - u * (t0 + dt / 2)) - g
        
        k4_v = f(t0 + dt, v0 + dt * k3_v)
        k4_a = (F - k * (v0 + dt * k3_v)**2) / float(m0 - u * (t0 + dt)) - g
        
        v0 = v0 + (dt / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        a0 = a0 + (dt / 6) * (k1_a + 2*k2_a + 2*k3_a + k4_a)
        t0 = t0 + dt
        
        t_values.append(t0)
        v_values.append(v0)
        a_values.append(a0)
    return t_values, v_values, a_values

h = 200000  # 步长
steps = int(t1 / h)

# 初始化速度和加速度
initial_velocity = 0.0
initial_acceleration = (F - k * initial_velocity**2) / float(m0) - g

# 求解速度和加速度
t_values, v_values, a_values = runge_kutta_4(velocity_prime_stage1, 0, initial_velocity, initial_acceleration, h, steps)

# 绘制时间与加速度的关系图
plt.figure(figsize=(10, 6))
plt.plot(t_values, a_values, label='Acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Time vs. Acceleration')
plt.legend()
plt.grid()
plt.show()

# 保存绘图
filename = "acceleration.png"
plt.savefig(filename)
plt.close()
