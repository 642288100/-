import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time  # 添加时间测量模块
import os  # 导入os模块

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
h_values = [0.02, 0.01, 0.005, 0.001]

# 存储五种方法的结果和时间
results_1 = []
results_2 = []
results_3 = []
results_4 = []
results_5 = []

execution_times_1 = []
execution_times_2 = []
execution_times_3 = []
execution_times_4 = []
execution_times_5 = []

# 计算数值解及误差（使用改进的欧拉法）
for h in h_values:
    num_steps = int((T - t0) / h)
    t_values = np.linspace(t0, T, num_steps + 1)
    u_values_1 = np.zeros(num_steps + 1)
    u_values_1[0] = u0

    start_time = time.time()  # 记录开始时间

    for i in range(num_steps):
        k1 = h * f(t_values[i], u_values_1[i])
        k2 = h * f(t_values[i] + 0.5 * h, u_values_1[i] + 0.5 * k1)  # 使用改进的欧拉法
        u_values_1[i+1] = u_values_1[i] + k2
    
    error = np.abs(u_values_1 - exact_solution(t_values))
    max_error = np.max(error)
    avg_error = np.mean(error)
    results_1.append((t_values, u_values_1, error, max_error, avg_error))  # 存储 max_error 和 avg_error

    end_time = time.time()  # 记录结束时间
    execution_times_3.append(end_time - start_time)  # 计算并存储时间

# 计算数值解及误差（使用4级4阶的Runge-Kutta法）
for h in h_values:
    num_steps = int((T - t0) / h)
    t_values = np.linspace(t0, T, num_steps + 1)
    u_values_2 = np.zeros(num_steps + 1)
    u_values_2[0] = u0
    
    start_time = time.time()  # 记录开始时间

    for i in range(num_steps):
        k1 = h * f(t_values[i], u_values_2[i])
        k2 = h * f(t_values[i] + 0.5 * h, u_values_2[i] + 0.5 * k1)
        k3 = h * f(t_values[i] + 0.5 * h, u_values_2[i] + 0.5 * k2)
        k4 = h * f(t_values[i] + h, u_values_2[i] + k3)
        
        u_values_2[i+1] = u_values_2[i] + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    error = np.abs(u_values_2 - exact_solution(t_values))
    max_error = np.max(error)
    avg_error = np.mean(error)
    results_2.append((t_values, u_values_2, error, max_error, avg_error))  # 存储 max_error 和 avg_error

    end_time = time.time()  # 记录结束时间
    execution_times_4.append(end_time - start_time)  # 计算并存储时间

    # 计算数值解及误差（使用梯形公式法）
for h in h_values:
    num_steps = int((T - t0) / h)
    t_values = np.linspace(t0, T, num_steps + 1)
    u_values_3 = np.zeros(num_steps + 1)
    u_values_3[0] = u0
    
    start_time = time.time()  # 记录开始时间

    for i in range(num_steps):
        k1 = h * f(t_values[i], u_values_3[i])
        k2 = h * f(t_values[i+1], u_values_3[i] + k1)
        u_values_3[i+1] = u_values_3[i] + 0.5 * (k1 + k2)
    
    error = np.abs(u_values_3 - exact_solution(t_values))
    max_error = np.max(error)
    avg_error = np.mean(error)
    results_3.append((t_values, u_values_3, error, max_error, avg_error))  # 存储 max_error 和 avg_error

    end_time = time.time()  # 记录结束时间
    execution_times_2.append(end_time - start_time)  # 计算并存储时间

# 计算数值解及误差(显示欧拉法)
for h in h_values:
    num_steps = int((T - t0) / h)
    t_values = np.linspace(t0, T, num_steps + 1)
    u_values_4 = np.zeros(num_steps + 1)
    u_values_4[0] = u0
    
    start_time = time.time()  # 记录开始时间

    for i in range(num_steps):
        u_values_4[i+1] = u_values_4[i] + h * f(t_values[i], u_values_4[i])
    
    error = np.abs(u_values_4 - exact_solution(t_values))
    max_error = np.max(error)
    avg_error = np.mean(error)
    results_4.append((t_values, u_values_4, error, max_error, avg_error))  # 存储 max_error 和 avg_error

    end_time = time.time()  # 记录结束时间
    execution_times_1.append(end_time - start_time)  # 计算并存储时间

# 计算数值解及误差（使用Adams-Bashforth方法）
for h in h_values:
    num_steps = int((T - t0) / h)
    t_values = np.linspace(t0, T, num_steps + 1)
    u_values_5 = np.zeros(num_steps + 1)
    u_values_5[:4] = [u0] + [exact_solution(t0 + i * h) for i in range(1, 4)]  # 使用精确解初始化前三步

    start_time = time.time()  # 记录开始时间

    for i in range(3, num_steps):
        u_predictor = u_values_5[i] + h / 24 * (55 * f(t_values[i], u_values_5[i])
                                              - 59 * f(t_values[i-1], u_values_5[i-1])
                                              + 37 * f(t_values[i-2], u_values_5[i-2])
                                              - 9 * f(t_values[i-3], u_values_5[i-3]))
        u_corrector = u_values_5[i] + h / 24 * (9 * f(t_values[i+1], u_predictor)
                                              + 19 * f(t_values[i], u_values_5[i])
                                              - 5 * f(t_values[i-1], u_values_5[i-1])
                                              + f(t_values[i-2], u_values_5[i-2]))
        u_values_5[i+1] = u_corrector
    
    error = np.abs(u_values_5 - exact_solution(t_values))
    max_error = np.max(error)
    avg_error = np.mean(error)
    results_5.append((t_values, u_values_5, error, max_error, avg_error))  # 存储 max_error 和 avg_error

    end_time = time.time()  # 记录结束时间
    execution_times_5.append(end_time - start_time)  # 计算并存储时间

# 输出计算时间结果
print("Execution times for different methods and step sizes:")
for i, h in enumerate(h_values):
    print(f"Step size: {h}")
    print(f"显式欧拉法: {execution_times_1[i]:.6f} seconds")
    print(f"梯形公式法: {execution_times_2[i]:.6f} seconds")
    print(f"改进的欧拉法: {execution_times_3[i]:.6f} seconds")
    print(f"Runge-Kutta 法: {execution_times_4[i]:.6f} seconds")
    print(f"dams-Bashforth方法: {execution_times_5[i]:.6f} seconds")
    print()

# 获取硬件条件并打印
print("Hardware Conditions:")
print("CPU:", os.cpu_count(), "cores")
print("RAM:", round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.0 ** 3), 2), "GB")

# 创建 DataFrame 存储计算时间结果
data = {
    'Step Size': h_values,
    '显式欧拉法': execution_times_1,
    '梯形公式法': execution_times_2,
    '改进的欧拉法': execution_times_3,
    'Runge-Kutta 法': execution_times_4,
    'Adams-Bashforth方法': execution_times_5
}

time_df = pd.DataFrame(data)

# 将计算时间结果保存到 Excel 文件
time_filename = "calculation_times.xlsx"
time_df.to_excel(time_filename, index=False)