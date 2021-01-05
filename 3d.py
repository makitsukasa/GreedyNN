import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

def sphere(x):
	return -np.sum(np.array(x) ** 2)

def ackley(x_):
	x = np.copy(x_) * 32.768
	return -20 + 20 * np.exp(- 0.2 * np.sqrt(1.0 / len(x) * np.sum(x ** 2)))\
		- np.e + np.exp(1.0 / len(x) * np.sum(np.cos(2 * np.pi * x)))

def rastrigin(x_):
	x = np.copy(x_) * 5.12
	return -10 * len(x) - np.sum(x ** 2) + 10 * np.sum(np.cos(2 * np.pi * x))

def rastrigin_offset(x_):
	x = (np.copy(x_) - 0.5) * 5.12
	return -10 * len(x) - np.sum(x ** 2) + 10 * np.sum(np.cos(2 * np.pi * x))

evaluator = rastrigin_offset

fig = plt.figure(figsize = (10, 6))
ax = fig.gca(projection='3d')

# answer curve
N = 500
x = np.linspace(-1, 1, N * 2).reshape(N, 2)
x = np.meshgrid(x[:, 0], x[:, 1])
y = np.array([[evaluator([x[0][i][j], x[1][i][j]]) for j in range(len(x[0][i]))] for i in range(len(x[0]))])
ax.plot_wireframe(x[0], x[1], y, color='blue', linewidth=0.1)

# sampling
N = 20
x = np.random.rand(N, 2) * 2 - 1
y = np.apply_along_axis(evaluator, 1, x)

# fit
# polyfit
p = np.zeros((x.shape[1], 3))
y_pred = np.zeros(x.shape[0])
for i in range(x.shape[1]):
	p[i, :] = np.polyfit(x[:, i], y, 2)
	y_pred += (p[i, 0] * x[:, i] ** 2 + p[i, 1] * x[:, i] + p[i, 2]) / x.shape[1]
y_gap = y - y_pred
y_gap_indices = np.argsort(y_gap)
ax.plot(x[:, 0], x[:, 1], y_pred, marker="o", linestyle='None')
ax.plot(x[y_gap_indices][-3:, 0], x[y_gap_indices][-3:, 1], y[y_gap_indices][-3:], marker="o", linestyle='None')
ax.plot(x[y_gap_indices][:-3, 0], x[y_gap_indices][:-3, 1], y[y_gap_indices][:-3], marker="o", linestyle='None')

# fit curve
# N = 500
# x = np.linspace(-1, 1, N * 2).reshape(N, 2)
# x = np.meshgrid(x[:, 0], x[:, 1])
# y_pred = np.zeros(N)
# for i in range(p.shape[1]):
# 	print(p[i, 0].shape)
# 	print(x[i].shape)
# 	y_pred += (p[i, 0] * x[i] ** 2 + p[i, 1] * x[i] + p[i, 2]) / p.shape[1]

plt.show()
input()
