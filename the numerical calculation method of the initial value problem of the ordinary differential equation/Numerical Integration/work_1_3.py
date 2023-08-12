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
h = 0.01

# 存储结果
results = []

# 计算数值解及误差（使用4级4阶的Runge-Kutta法）

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

#使用复化梯形公式计算积分：
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    integral = h * (np.sum(y) - 0.5 * (y[0] + y[-1]))
    return integral

integral_approximation = trapezoidal_rule(lambda t: np.interp(t, t_values, u_values), 0, T, num_steps)

#计算精确积分：
def exact_integral(t):
    return np.exp(t) * (np.cos(t) + np.sin(t))

integral_exact = exact_integral(T) - exact_integral(t0)

#输出结果：
print("Approximation using Runge-Kutta method:", u_values[-1])
print("Approximation using Trapezoidal Rule:", integral_approximation)
print("Exact Integral:", integral_exact)
