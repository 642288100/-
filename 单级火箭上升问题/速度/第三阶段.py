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

# 第三阶段速度函数
def velocity_prime_stage2(_, v):
    denominator = m
    if denominator == 0:
        print("分母为0")
    return (k * v**2) / float(denominator) - g

# 使用经典四阶Runge-Kutta方法求解第三阶段速度
def runge_kutta_4_stage2(f, t_start, v_start, dt, steps):
    t_values = [t_start]
    v_values = [v_start]
    for _ in range(steps):
        k1 = f(None, v_start)  # 第二阶段速度函数不依赖于时间
        k2 = f(None, v_start + dt / 2 * k1)
        k3 = f(None, v_start + dt / 2 * k2)
        k4 = f(None, v_start + dt * k3)
        
        v_start = v_start + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        t_start = t_start + dt
        
        t_values.append(t_start)
        v_values.append(v_start)
        

    return v_values, t_values

h = 20000000
t3 = 125.1 #估算第三阶段结束时的时间

# 使用第二阶段结束时的速度 v_t2 作为第三阶段的初始速度
v_values, t_values= runge_kutta_4_stage2(velocity_prime_stage2, t2, v_t2, (t3 - t2) / h, h)

# 输出第三阶段结束时的速度
print("第三阶段结束时的速度：", v_values[-1])

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