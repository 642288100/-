import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 定义微分方程和精确解
def f(t, u):
    return -99 * u + np.exp(t) * (100 * np.cos(t) - np.sin(t))

def exact_solution(t):
    return np.exp(t) * np.cos(t)

# 参数设置
t0 = 0.0
T = 2.0
u0 = 1.0

# 不同步长
h_values = [0.008, 0.004, 0.002, 0.001]

# 存储结果
results = []

# 计算数值解及误差（使用4级4阶的Runge-Kutta法）
for h in h_values:
    num_steps = int((T - t0) / h)
    t_values = np.linspace(t0, T, num_steps + 1)
    u_values = np.zeros(num_steps + 1)
    u_values[0] = u0
    
    for i in range(num_steps):
        k1 = h * f(t_values[i], u_values[i])
        k2 = h * f(t_values[i] + 0.5 * h, u_values[i] + 0.5 * k1)
        k3 = h * f(t_values[i] + 0.5 * h, u_values[i] + 0.5 * k2)
        k4 = h * f(t_values[i] + h, u_values[i] + k3)
        
        u_values[i+1] = u_values[i] + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    error = np.abs(u_values - exact_solution(t_values))
    max_error = np.max(error)
    avg_error = np.mean(error)
    results.append((t_values, u_values, error, max_error, avg_error))  # 存储 max_error 和 avg_error


# 分段Lagrange插值函数
def piecewise_lagrange_interpolation(x, x_values, y_values):
    n = len(x_values)
    result = 0.0
    
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if i != j:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        result += term
    
    return result

# 待插值点
interpolation_points = [0.97, 1.03, 1.51, 1.96]

# 进行插值计算
interpolated_values = []
for x in interpolation_points:
    interval_index = int((x - t0) / h)
    interval_x_values = t_values[interval_index:interval_index+2]
    interval_y_values = u_values[interval_index:interval_index+2]
    interpolated_value = piecewise_lagrange_interpolation(x, interval_x_values, interval_y_values)
    interpolated_values.append(interpolated_value)

# 输出插值结果
for i, x in enumerate(interpolation_points):
    print(f"Interpolated u({x:.2f}): {interpolated_values[i]:.6f}")




