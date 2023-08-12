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
h_values = [0.02, 0.01, 0.005, 0.001]


# 存储不同方法的结果
results_1 = []
results_2 = []
results_3 = []
results_4 = []
results_5 = []

# 计算数值解及误差（使用改进的欧拉法）
for h in h_values:
    num_steps = int((T - t0) / h)
    t_values = np.linspace(t0, T, num_steps + 1)
    u_values_1 = np.zeros(num_steps + 1)
    u_values_1[0] = u0
    
    for i in range(num_steps):
        k1 = h * f(t_values[i], u_values_1[i])
        k2 = h * f(t_values[i] + 0.5 * h, u_values_1[i] + 0.5 * k1)  # 使用改进的欧拉法
        u_values_1[i+1] = u_values_1[i] + k2
    
    error = np.abs(u_values_1 - exact_solution(t_values))
    max_error = np.max(error)
    avg_error = np.mean(error)
    results_1.append((t_values, u_values_1, error, max_error, avg_error))  # 存储 max_error 和 avg_error

# 计算数值解及误差（使用4级4阶的Runge-Kutta法）
for h in h_values:
    num_steps = int((T - t0) / h)
    t_values = np.linspace(t0, T, num_steps + 1)
    u_values_2 = np.zeros(num_steps + 1)
    u_values_2[0] = u0
    
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

# 计算数值解及误差（使用梯形公式法）
for h in h_values:
    num_steps = int((T - t0) / h)
    t_values = np.linspace(t0, T, num_steps + 1)
    u_values_3 = np.zeros(num_steps + 1)
    u_values_3[0] = u0
    
    for i in range(num_steps):
        k1 = h * f(t_values[i], u_values_3[i])
        k2 = h * f(t_values[i+1], u_values_3[i] + k1)
        u_values_3[i+1] = u_values_3[i] + 0.5 * (k1 + k2)
    
    error = np.abs(u_values_3 - exact_solution(t_values))
    max_error = np.max(error)
    avg_error = np.mean(error)
    results_3.append((t_values, u_values_3, error, max_error, avg_error))  # 存储 max_error 和 avg_error

# 计算数值解及误差(显示欧拉法)
for h in h_values:
    num_steps = int((T - t0) / h)
    t_values = np.linspace(t0, T, num_steps + 1)
    u_values_4 = np.zeros(num_steps + 1)
    u_values_4[0] = u0
    
    for i in range(num_steps):
        u_values_4[i+1] = u_values_4[i] + h * f(t_values[i], u_values_4[i])
    
    error = np.abs(u_values_4 - exact_solution(t_values))
    max_error = np.max(error)
    avg_error = np.mean(error)
    results_4.append((t_values, u_values_4, error, max_error, avg_error))  # 存储 max_error 和 avg_error

# 计算数值解及误差（使用Adams-Bashforth方法）
for h in h_values:
    num_steps = int((T - t0) / h)
    t_values = np.linspace(t0, T, num_steps + 1)
    u_values_5 = np.zeros(num_steps + 1)
    u_values_5[:4] = [u0] + [exact_solution(t0 + i * h) for i in range(1, 4)]  # 使用精确解初始化前三步

    
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

'''
# 绘制图像
plt.figure(figsize=(12, 10))

for j in range(4):
    plt.subplot(2, 2, j + 1)

    plt.plot(t_values, u_values_1, label=f'Improved Euler method (Step size: {h_values[j]})', linewidth=8,linestyle='dashdot',alpha=0.7)

    plt.plot(t_values, u_values_2, label=f'Runge-Kutta method (Step size: {h_values[j]})', linewidth=6.5, linestyle='dotted',alpha=0.7)

    plt.plot(t_values, u_values_3, label=f'Trapezoidal rule method (Step size: {h_values[j]})', linewidth=5,alpha=0.7)

    plt.plot(t_values, u_values_4, label=f'Explicit Euler method (Step size: {h_values[j]})', linewidth=3.5,alpha=0.7)

    plt.plot(t_values, u_values_5, label=f'Adams—Bashforth method (Step size: {h_values[j]})', linewidth=2,alpha=0.7)

    plt.plot(t_values, exact_solution(t_values), label='Exact', linewidth=1.5, linestyle='dashed', color='black')
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.legend()
    plt.title(f'Step size: {h_values[j]}')  # 在子图上添加步长信息

plt.suptitle('Numerical Solutions vs. Exact Solution', fontsize=16, y=0.98)  # 在全图上添加标题
plt.tight_layout()
plt.show()  # 显示图像

# 保存图像
filename = "all_methods_visualization.png"
plt.savefig(filename)
plt.show()
plt.close()
'''

# 创建一个2x2的子图，绘制最大误差
plt.figure(figsize=(12, 10))

for i, h in enumerate(h_values):
    plt.subplot(2, 2, i + 1)
    
    max_error_lines = []
    for j, result in enumerate([results_1, results_2, results_3, results_4, results_5]):
        max_error_line, = plt.plot(h_values, [res[3] for res in result], marker='o', label=f'Max Error method_labels({[j]})', linewidth=2.5)
        max_error_lines.append(max_error_line)
    
    plt.xlabel('Step Size')
    plt.ylabel('Max Error')
    plt.yscale('log')  # 设置对数坐标轴
    
    plt.title(f'Step Size: {h}')
    
    # 创建图例
    max_error_labels = [line.get_label() for line in max_error_lines]
    plt.legend(handles=max_error_lines, labels=max_error_labels, loc='upper left')

# 调整子图布局
plt.tight_layout()

# 保存图像
filename_max_error = "max_errors_subplots_methods.png"
plt.savefig(filename_max_error)
plt.show()


# 创建一个2x2的子图，绘制平均误差
plt.figure(figsize=(12, 10))

for i, h in enumerate(h_values):
    plt.subplot(2, 2, i + 1)
    
    avg_error_lines = []
    for j, result in enumerate([results_1, results_2, results_3, results_4, results_5]):
        avg_error_line, = plt.plot(h_values, [res[4] for res in result], marker='x', label=f'Avg Error method_labels({[j]})', linewidth=1.5)
        avg_error_lines.append(avg_error_line)
    
    plt.xlabel('Step Size')
    plt.ylabel('Avg Error')
    plt.yscale('log')  # 设置对数坐标轴
    
    plt.title(f'Step Size: {h}')
    
    # 创建图例
    avg_error_labels = [line.get_label() for line in avg_error_lines]
    plt.legend(handles=avg_error_lines, labels=avg_error_labels, loc='upper left')

# 调整子图布局
plt.tight_layout()

# 保存图像
filename_avg_error = "avg_errors_subplots_methods.png"
plt.savefig(filename_avg_error)
plt.show()
