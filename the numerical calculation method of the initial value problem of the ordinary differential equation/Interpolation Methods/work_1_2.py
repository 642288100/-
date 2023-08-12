import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, PchipInterpolator
from sympy import symbols, diff
from scipy.interpolate import lagrange
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

# 步长
h = 0.01

# 计算数值解
num_steps = int((T - t0) / h)
t_values = np.linspace(t0, T, num_steps + 1)
u_values = np.zeros(num_steps + 1)
u_values[0] = u0

for i in range(num_steps):
    k1 = h * f(t_values[i], u_values[i])
    k2 = h * f(t_values[i] + 0.5 * h, u_values[i] + 0.5 * k1)
    k3 = h * f(t_values[i] + 0.5 * h, u_values[i] + 0.5 * k2)
    k4 = h * f(t_values[i] + h, u_values[i] + k3)
    
    u_values[i+1] = u_values[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

# 插值点
interpolation_points = [0.97, 1.03, 1.51, 1.96]

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

# 进行插值计算
interpolated_values = []
for x in interpolation_points:
    interval_index = int((x - t0) / h)
    interval_x_values = t_values[interval_index:interval_index+2]
    interval_y_values = u_values[interval_index:interval_index+2]
    interpolated_value = piecewise_lagrange_interpolation(x, interval_x_values, interval_y_values)
    interpolated_values.append(interpolated_value)

# 分段Hermite插值
hermite_results = []
for point in interpolation_points:
    hermite_interp = PchipInterpolator(t_values, u_values) #
    hermite_result = hermite_interp(point)
    hermite_results.append(hermite_result)

# 输出结果
for i in range(len(interpolation_points)):
    exact_result = exact_solution(interpolation_points[i])
    print(f"Interpolation Point: {interpolation_points[i]}")
    print(f"Exact Solution: {exact_result:.6f}")
    print(f"Lagrange Interpolation: {interpolated_values[i]:.6f}")
    print(f"Hermite Interpolation: {hermite_results[i]:.6f}")
    print()


# 创建一个空的DataFrame来存储插值结果
result_data = []

# 计算并存储插值结果
for i in range(len(interpolation_points)):
    exact_result = exact_solution(interpolation_points[i])
    lagrange_interp_result = interpolated_values[i]
    hermite_interp_result = hermite_results[i]
    
    result_data.append({
        'Interpolation Point': interpolation_points[i],
        'Exact Solution': exact_result,
        'Lagrange Interpolation': lagrange_interp_result,
        'Hermite Interpolation': hermite_interp_result
    })

# 创建DataFrame
result_df = pd.DataFrame(result_data)

# 将结果保存到Excel文件
excel_filename = 'interpolation_results.xlsx'
result_df.to_excel(excel_filename, index=False)
print(f"Interpolation results saved to {excel_filename}.")

# 绘制图像
# plt.plot(t_values, u_values, label='Numerical') # 绘制数值解
plt.plot(t_values, exact_solution(t_values), label='Exact')
plt.scatter(interpolation_points, hermite_results, color='orange', label='Hermite Interpolation')
plt.scatter(interpolation_points, interpolated_values, color='red',label='Lagrange Interpolation',marker='x') # 绘制插值点
plt.xlabel('t')
plt.ylabel('u(t)')
plt.legend()
plt.title('Interpolation Comparison')
plt.show()

#保存图像   
filename = "interpolation_comparison.png"
plt.savefig(filename)
plt.close()