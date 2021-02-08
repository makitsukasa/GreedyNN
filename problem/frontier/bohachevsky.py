import numpy as np

def bohachevsky(x):
	shifted = x * 10.24 - 5.12
	a = shifted[:-1] ** 2
	b = 2 * shifted[1:] ** 2
	c = 0.3 * np.cos(3 * np.pi * shifted[:-1])
	d = 0.4 * np.cos(4 * np.pi * shifted[1:])
	return -np.sum(a + b - c - d + 0.7)

def bohachevsky_offset(x):
	shifted = x * 10.24 - 5.12
	a = shifted[:-1] ** 2
	b = 2 * shifted[1:] ** 2
	c = 0.3 * np.cos(3 * np.pi * shifted[:-1])
	d = 0.4 * np.cos(4 * np.pi * shifted[1:])
	return -np.sum(a + b - c - d + 0.7)
