import numpy as np

def ackley(x_):
	x = x_ * 32.768
	n = len(x)
	return -20 + 20 * np.exp(-0.2 * np.sqrt(1.0 * np.sum(x ** 2)) / n)\
		- np.e + np.exp(np.sum(np.cos(2 * np.pi * x)) / n)

def ackley_offset(x_):
	x = (x_ - 0.5) * 32.768
	n = len(x)
	return -20 + 20 * np.exp(-0.2 * np.sqrt(1.0 * np.sum(x ** 2)) / n)\
		- np.e + np.exp(np.sum(np.cos(2 * np.pi * x)) / n)

def ackley_offset3(x_):
	x = (x_ - 0.3) * 32.768
	n = len(x)
	return -20 + 20 * np.exp(-0.2 * np.sqrt(1.0 * np.sum(x ** 2)) / n)\
		- np.e + np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
