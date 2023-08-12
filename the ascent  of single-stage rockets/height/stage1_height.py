import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

# 参数设置
m0 = 1200  # 初始质量（kg）
u = 15     # 燃料消耗速率（kg/s）
F = 27000   # 推力（N）
k = 0.4    # 空气阻力比例系数（kg/m）
g = 9.8    # 重力加速度（m/s^2）
t1 = 600 / u  # 引擎关闭时间（s）

# 第一阶段速度函数
def velocity_prime_stage1(t, v):
    denominator = m0 - u * t
    if denominator == 0:
        print("分母为0")  # 或者返回其他合适的值，避免除以零
    return (F - k * v**2) / float(denominator) - g
# 数值积分方法
def integrate(func, t_values, initial_conditions):
    v_values = [initial_conditions[1]]
    for i in range(1, len(t_values)):
        dt = t_values[i] - t_values[i-1]
        v_next = v_values[-1] + func(t_values[i-1], v_values[-1]) * dt
        v_values.append(v_next)
    return v_values

# 时间步长和时间点
dt = t1/20000000
t_values = np.arange(0, t1, dt)

# 初始条件
v0 = 0  # 初始速度为0

# 数值积分得到速度随时间变化的数据
v_values = integrate(velocity_prime_stage1, t_values, (t_values[0], v0))

# 数值积分得到高度随时间变化的数据
h_values = np.cumsum(v_values) * dt

print("第一阶段结束时的高度：", h_values[-1])

# 绘制时间与高度关系图
plt.plot(t_values, h_values, label='stage1_height')
plt.xlabel('time (s)')
plt.ylabel('height (m)')
plt.title('Rocket Flight: Time vs Height')
plt.legend()
plt.grid(True)
plt.show()


#保存绘图
filename = "velocity_stage1_height.png"
plt.savefig(filename)
plt.close()