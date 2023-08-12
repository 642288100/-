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

# 计算加速度数据
# 设置阀值
threshold = 0.1
# 计算加速度数据并处理突增情况
acceleration_values = []
for i in range(len(t_values)):
    acceleration = velocity_prime_stage1(t_values[i], v_values[i])
    
    # 处理突增情况
    if i > 0 and abs(acceleration - acceleration_values[i - 1]) > threshold:  # 设定一个阈值，来判断是否为突增
        small_decrease = 0.01  # 可以调整微小减少量的大小
        adjusted_acceleration = acceleration_values[i - 1] - small_decrease
        acceleration_values.append(adjusted_acceleration)
    else:
        acceleration_values.append(acceleration)

# i每1000个数据取一个
# for i in range(0, len(t_values), 10000):
#   print(acceleration_values[i])

print("第一阶段开始时的加速度：", acceleration_values[0])
print("第一阶段结束加速度的值：", acceleration_values[-1])

# 绘制加速度与时间的关系图
plt.plot(t_values, acceleration_values, label="acceleration_stage1")
plt.xlabel("time (s)")
plt.ylabel("acceleration (m/s^2)")
plt.title("Acceleration_stage1 VS Time (Runge-Kutta method)")
plt.grid(True)
plt.legend()
plt.show()

# 保存绘图
plt.savefig("acceleration_stage1.png", dpi=500)