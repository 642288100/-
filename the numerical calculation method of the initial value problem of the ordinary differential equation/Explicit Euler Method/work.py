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

# 存储结果
results = []

# 计算数值解及误差(显示欧拉法)
for h in h_values:
    num_steps = int((T - t0) / h)
    t_values = np.linspace(t0, T, num_steps + 1)
    u_values = np.zeros(num_steps + 1)
    u_values[0] = u0
    
    for i in range(num_steps):
        u_values[i+1] = u_values[i] + h * f(t_values[i], u_values[i])
    
    error = np.abs(u_values - exact_solution(t_values))
    max_error = np.max(error)
    avg_error = np.mean(error)
    results.append((t_values, u_values, error, max_error, avg_error))  # 存储 max_error 和 avg_error

# 输出结果
for i, (t_values, u_values, error, max_error, avg_error) in enumerate(results):
    print(f"Step size: {h_values[i]}")
    print(f"Max error: {max_error:.6f}")
    print(f"Avg error: {avg_error:.6f}")
    print()

# 将结果保存到Excel表格
data = []

for i, (t_values, u_values, error, max_error, avg_error) in enumerate(results):
    data.append({
        'Step size': h_values[i],
        'Max error': max_error,
        'Avg error': avg_error
    })

df = pd.DataFrame(data)
df.to_excel('results.xlsx', index=False)

print("Results saved to results.xlsx.")

# 绘制图像
plt.figure(figsize=(12, 8))

for i, (t_values, u_values, _, _, _) in enumerate(results):
    plt.subplot(2, 2, i + 1)
    plt.plot(t_values, u_values, label='Numerical',linewidth=5)
    plt.plot(t_values, exact_solution(t_values), label='Exact',linewidth=2.5)
    plt.title(f'Step size: {h_values[i]}')
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.legend()

plt.tight_layout()

# 保存图像
filename = "visualization.png"
plt.savefig(filename)
plt.close()


# 绘制最大误差图（左纵轴）
max_error_line, =plt.plot(h_values, [result[3] for result in results], marker='o', label='Max Error',markersize=10,linewidth=2.5)
plt.xlabel('Step Size')
plt.ylabel('Max Error')
plt.yscale('log')  # 设置对数坐标轴

# 创建一个共享横轴的子图，用于绘制平均误差图（右纵轴）
ax2 = plt.gca().twinx() # gca()函数返回当前坐标轴
avg_error_line, =ax2.plot(h_values, [result[4] for result in results], marker='x', color='orange', label='Avg Error',markersize=8,linewidth=1.5)
ax2.set_ylabel('Avg Error')
ax2.set_yscale('log') # 设置对数坐标轴


plt.legend(handles=[max_error_line, avg_error_line], loc='upper left')

plt.title('Max Error(Log Scale) and Avg Error(Log Scale) vs. Step Size')
plt.tight_layout()

# 保存图像
filename = "errors_combined.png"
plt.savefig(filename)
plt.show()
plt.close()





