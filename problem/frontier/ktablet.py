import numpy as np

def ktablet(x):
	shifted = x * 10.24
	k = len(x) // 4
	return -np.sum(shifted[:k] ** 2) - np.sum((100.0 * shifted[k:]) ** 2)

def ktablet_offset(x):
	shifted = (x - 0.5) * 10.24
	k = len(x) // 4
	return -np.sum(shifted[:k] ** 2) - np.sum((100.0 * shifted[k:]) ** 2)
