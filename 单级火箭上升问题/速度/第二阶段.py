import numpy as np
import matplotlib.pyplot as plt

# 参数设置
m0 = 1200  # 初始质量（kg）
m1 = 1100   # 燃料的质量（kg）
m = 100    # 第二阶段初始质量（kg）
u = 15     # 燃料消耗速率（kg/s）
F = 27000  # 推力（N）
k = 0.4    # 空气阻力比例系数（kg/m）
g = 9.8    # 重力加速度（m/s^2）
t1 = 1100 / u  # 引擎关闭时间（s）80
v_t1 = 254.66626485855937
#v_t1 = 226.67275810501403  # 引擎关闭时刻的速度
# 第二阶段结束时的时间： 53.35699200078852

# 第二阶段速度函数
def velocity_prime_stage2(_, v):
    denominator = m
    if denominator == 0:
        print("分母为0")
    return (-k * v**2) / float(denominator) - g

# 使用经典四阶Runge-Kutta方法求解第二阶段速度
def runge_kutta_4_stage2(f, t_start, v_start, dt, steps):
    t_values = [t_start]
    v_values = [v_start]
    t_end=0
    for _ in range(steps):
        k1 = f(None, v_start)  # 第二阶段速度函数不依赖于时间
        k2 = f(None, v_start + dt / 2 * k1)
        k3 = f(None, v_start + dt / 2 * k2)
        k4 = f(None, v_start + dt * k3)
        
        v_start = v_start + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        t_start = t_start + dt
        
        t_values.append(t_start)
        v_values.append(v_start)
        
        if v_start <= 0 :
            t_end = t_start
            break

    return v_values, t_values, t_end

h = 20000000
t2 = 95.0

# 使用第一阶段结束时的速度 v_t1 作为第二阶段的初始速度
v_values, t_values,t_end = runge_kutta_4_stage2(velocity_prime_stage2, t1, v_t1, (t2 - t1) / h, h)

# 输出第二阶段结束时的速度
print("第二阶段结束时的速度：", v_values[-1])
print("第二阶段结束时的时间：", t_end)

'''
# 绘图
plt.plot(t_values, v_values)
plt.xlabel('time (s)')
plt.ylabel('speed (m/s)')
plt.title('Rocket Second Stage Velocity Variation (Runge-Kutta method)')
plt.grid(True)
plt.show()

# 保存绘图
filename = "velocity_stage2.png"
plt.savefig(filename)
plt.close()
'''
