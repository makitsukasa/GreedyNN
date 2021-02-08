import numpy as np

def ellipsoid(x):
	shifted = x * 10.24 - 5.12
	n = len(x)
	ret = 0
	for i in range(n):
		# ret += 10 ** (6.0 * i / (n - 1)) * shifted[i] ** 2
		# ret += (1000 ** (i * 1.0 / (n - 1)) * shifted[i]) ** 2
		ret += np.power(np.power(1000, float(i) / (n - 1)) * shifted[i], 2)
	return ret
