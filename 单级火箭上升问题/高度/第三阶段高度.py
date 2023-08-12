import numpy as np
import matplotlib.pyplot as plt


# 参数设置
m0 = 1200  # 初始质量（kg）
m1 = 600   # 燃料的质量（kg）
m = 600    # 第二阶段初始质量（kg）
u = 15     # 燃料消耗速率（kg/s）
F = 27000  # 推力（N）
k = 0.4    # 空气阻力比例系数（kg/m）
g = 9.8    # 重力加速度（m/s^2）
t1 = 600 / u  # 引擎关闭时间（s）80
v_t1 = 226.67275810501403  # 引擎关闭时刻的速度
v_t2 = 0 # 第二阶段结束时的速度
t2 = 53.35699200078852 #第二阶段结束时的时间
h2 = 7653.799878685426 #第二阶段结束时的高度

# 第三阶段速度函数
def velocity_prime_stage2(_, v):
    denominator = m
    if denominator == 0:
        print("分母为0")
    return np.float128(k * v**2) / np.float128(denominator) - g

# 数值积分方法
def integrate(func, t_values, initial_conditions):
    v_values = [initial_conditions[1]]
    for i in range(1, len(t_values)):
        dt = t_values[i] - t_values[i-1]
        v_next = v_values[-1] + func(t_values[i-1], v_values[-1]) * dt
        v_values.append(v_next)
    return v_values

t3 = 125.1 #估算第三阶段结束时的时间

# 时间步长和时间点
dt = float((t3-t2)/10000000)
t_values = np.arange(t2, t3, dt)

# 初始条件
v0= v_t2  # 初始速度为v_t2

# 数值积分得到速度随时间变化的数据
v_values = integrate(velocity_prime_stage2, t_values, (t_values[0], v0))

# 数值积分得到高度随时间变化的数据
h_values = h2 + np.cumsum(v_values) * dt



print("第三阶段结束时的高：", h_values[-1])
print("第三阶段结束时的速度：", v_values[-1])

# 绘制时间与高度关系图
plt.plot(t_values, h_values, label='stage3_height')
plt.xlabel('time (s)')
plt.ylabel('height (m)')
plt.title('Rocket Flight: Time vs Height')
plt.legend()
plt.grid(True)
plt.show()


#保存绘图
filename = "velocity_stage3_height.png"
plt.savefig(filename)
plt.close()

# 绘图
plt.plot(t_values, v_values)
plt.xlabel('time (s)')
plt.ylabel('speed (m/s)')
plt.title('Rocket Third Stage Velocity Variation (Runge-Kutta method)')
plt.grid(True)
plt.show()

# 保存绘图
filename = "velocity_stage3.png"
plt.savefig(filename)
plt.close()