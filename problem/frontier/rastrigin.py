import numpy as np

def rastrigin(x_):
	x = x_ * 5.12
	return -10 * len(x) - np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def rastrigin_offset(x_):
	x = (x_ - 0.5) * 5.12
	return -10 * len(x) - np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
