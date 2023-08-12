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
h_values = [0.012, 0.01, 0.005, 0.001]

# 存储结果
results = []

# 计算数值解及误差（使用Adams-Bashforth方法）
for h in h_values:
    num_steps = int((T - t0) / h)
    t_values = np.linspace(t0, T, num_steps + 1)
    u_values = np.zeros(num_steps + 1)
    u_values[:4] = [u0] + [exact_solution(t0 + i * h) for i in range(1, 4)]  # 使用精确解初始化前三步

    
    for i in range(3, num_steps):
        u_predictor = u_values[i] + h / 24 * (55 * f(t_values[i], u_values[i])
                                              - 59 * f(t_values[i-1], u_values[i-1])
                                              + 37 * f(t_values[i-2], u_values[i-2])
                                              - 9 * f(t_values[i-3], u_values[i-3]))
        u_corrector = u_values[i] + h / 24 * (9 * f(t_values[i+1], u_predictor)
                                              + 19 * f(t_values[i], u_values[i])
                                              - 5 * f(t_values[i-1], u_values[i-1])
                                              + f(t_values[i-2], u_values[i-2]))
        u_values[i+1] = u_corrector
    
    error = np.abs(u_values - exact_solution(t_values))
    max_error = np.max(error)
    avg_error = np.mean(error)
    results.append((t_values, u_values, error, max_error, avg_error))  # 存储 max_error 和 avg_error

# 将结果保存到Excel表格
data = []

for i, (t_values, u_values, error, max_error, avg_error) in enumerate(results):
    data.append({
        'Step size': h_values[i],
        'Max error': max_error,
        'Avg error': avg_error
    })

df = pd.DataFrame(data)
df.to_excel('results_adams_bashforth.xlsx', index=False)

print("Results saved to results_adams_bashforth.xlsx.")

# 绘制图像
plt.figure(figsize=(12, 8))

for i, (t_values, u_values, _, _, _) in enumerate(results):
    plt.subplot(2, 2, i + 1)
    plt.plot(t_values, u_values, label='Numerical', linewidth=5)
    plt.plot(t_values, exact_solution(t_values), label='Exact', linewidth=2.5)
    plt.title(f'Step size: {h_values[i]}')
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.legend()

plt.tight_layout()

# 保存图像
filename = "visualization_adams_bashforth.png"
plt.savefig(filename)
plt.close()

# 绘制最大误差图（左纵轴，对数坐标轴）
max_error_line, = plt.plot(h_values, [result[3] for result in results], marker='o', markersize=10, linewidth=2.5, label='Max Error')
plt.xlabel('Step Size')
plt.ylabel('Max Error')
plt.yscale('log')  # 设置对数坐标轴

# 创建一个共享横轴的子图，用于绘制平均误差图（右纵轴，对数坐标轴）
ax2 = plt.gca().twinx()
avg_error_line, = ax2.plot(h_values, [result[4] for result in results], marker='x', markersize=8, linewidth=1.5, color='orange', label='Avg Error')
ax2.set_ylabel('Avg Error')
ax2.set_yscale('log')  # 设置对数坐标轴

# 创建自定义图例
plt.legend(handles=[max_error_line, avg_error_line], loc='upper left')

plt.title('Max Error (Log Scale) and Avg Error (Log Scale) vs. Step Size')
plt.tight_layout()

# 保存图像
filename = "errors_log_scale_adams_bashforth.png"
plt.savefig(filename)
plt.show()
