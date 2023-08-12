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
v_t1 = 226.67275810501403 # 引擎关闭时刻的速度
t2 = 53.35699200078852

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
            break

    return v_values, t_values

h = 20000000

# 求解第一阶段的速度
t_values_1, v_values_1 = runge_kutta_4(velocity_prime_stage1, 0, 0, t1 / h, h)  # 使用1000倍的步数来更精确地近似

# 使用第一阶段结束时的速度 v_t1 作为第二阶段的初始速度
v_values_2, t_values_2 = runge_kutta_4_stage2(velocity_prime_stage2, t1, v_t1, (t2 - t1) / h, h)

# 绘图
stage1,=plt.plot(t_values_1, v_values_1,label='Stage 1')
stage2,=plt.plot(t_values_2, v_values_2,color='orange',label='Stage 2')
plt.xlabel('time (s)')
plt.ylabel('speed (m/s)')
plt.legend(handles=[stage1, stage2], loc='lower center')
plt.title('Rocket all Stage Velocity Variation(Runge-Kutta method)')
plt.grid(True)
plt.show()

#保存绘图
filename = "velocity_stage.png"
plt.savefig(filename)
plt.close()

