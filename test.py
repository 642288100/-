import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, PchipInterpolator
from sympy import symbols, diff

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

for t_value in t_values:
    print(t_value)
 
 #绘制exact_solution(t)的图像
plt.plot(t_values, exact_solution(t_values), label='Exact')
plt.xlabel('t')
plt.ylabel('u(t)')
plt.legend()
plt.title('Exact Solution')

#保存图像
filename = "exact_solution.png"
plt.savefig(filename)
plt.close()

