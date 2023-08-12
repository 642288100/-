import numpy as np
import matplotlib.pyplot as plt

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

# 改进的欧拉方法
def improved_euler_step(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h, y + k1)
    return y + 0.5 * (k1 + k2)

# 数值求解第一阶段速度
def solve_velocity_stage1():
    h = 0.000209  # 时间步长
    num_steps = int(t1 / h)
    t_values = np.linspace(0, t1, num_steps + 1)
    v_values = np.zeros(num_steps + 1)
    v_values[0] = 0

    for i in range(num_steps):
        t = t_values[i]
        v = v_values[i]
        v_values[i+1] = improved_euler_step(velocity_prime_stage1, t, v, h)


    return t_values, v_values

'''
# 数值求解第一阶段速度
def solve_velocity_stage1():
    h = 0.0001  # 时间步长
    num_steps = int(t1 / h)
    t_values = np.linspace(0, t1, num_steps + 1)
    v_values = np.zeros(num_steps + 1)
    v_values[0] = 0
    
    for i in range(num_steps):
        t = t_values[i]
        v = v_values[i]
        k1 = h * velocity_prime_stage1(t, v)
        k2 = h * velocity_prime_stage1(t + h / 2, v + k1 / 2)
        k3 = h * velocity_prime_stage1(t + h / 2, v + k2 / 2)
        k4 = h * velocity_prime_stage1(t + h, v + k3)
        v_values[i+1] = v + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t_values, v_values

'''

# 计算引擎关闭时刻的速度
t1_index = int(t1 / 0.000209)
v_t1 = solve_velocity_stage1()[1][t1_index]
print("引擎关闭时刻的速度为：", v_t1)

'''
# 第二阶段速度函数
def velocity_prime_stage2(t, v):
    return -k * v**2 / (m0 - m1) - g

# 数值求解第二阶段速度、高度和加速度
def solve_stage2(t_values, v0):
    m1 = u * (t1 - t_values[0])  # 质量减少量
    v_values = np.zeros_like(t_values)
    h_values = np.zeros_like(t_values)
    a_values = np.zeros_like(t_values)
    
    v_values[0] = v0
    h_values[0] = 0
    a_values[0] = velocity_prime_stage2(t_values[0], v0)
    
    for i in range(1, len(t_values)):
        t = t_values[i]
        v = v_values[i-1]
        h = h_values[i-1]
        a = a_values[i-1]
        
        v_prime = velocity_prime_stage2(t, v)
        h_prime = v
        a_prime = v_prime
        
        v_values[i] = v + v_prime * (t - t_values[i-1])
        h_values[i] = h + h_prime * (t - t_values[i-1])
        a_values[i] = a + a_prime * (t - t_values[i-1])
    
    return v_values, h_values, a_values

# 第一阶段速度和时间
t_values_stage1, v_values_stage1 = solve_velocity_stage1()

# 第二阶段时间
t2 = t_values_stage1[-1] + (v_t1 / g)

# 总时间和时间步长
total_time = t2
h = 0.01
num_steps = int(total_time / h)
t_values = np.linspace(t1, total_time, num_steps + 1)

# 第二阶段速度、高度和加速度
v_values_stage2, h_values_stage2, a_values_stage2 = solve_stage2(t_values, v_t1)

# 绘制图像
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(np.concatenate((t_values_stage1, t_values)), np.concatenate((v_values_stage1, v_values_stage2)))
plt.title('Rocket Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')

plt.subplot(3, 1, 2)
plt.plot(np.concatenate((t_values_stage1, t_values)), np.concatenate((h_values_stage1, h_values_stage2)))
plt.title('Rocket Height')
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')

plt.subplot(3, 1, 3)
plt.plot(np.concatenate((t_values_stage1, t_values)), np.concatenate((a_values_stage1, a_values_stage2)))
plt.title('Rocket Acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')

plt.tight_layout()
plt.show()
'''