# SHCLVND
# https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=CD5ECB7AC318EEE9EA632D3B57BCFB3B?doi=10.1.1.19.3536&rep=rep1&type=pdf

import numpy as np

class SHCLVND:
	def __init__(self, n_dim, evaluator, dist_mean_move = 0.05, dist_stddev_reduce_rate = None, elite_size = 3):
		self.n_dim = n_dim
		self.evaluator = evaluator
		self.dist_mean_move = dist_mean_move
		self.dist_stddev_reduce_rate = dist_stddev_reduce_rate
		self.elite_size = elite_size

	def train(self, n_generation, n_child):
		dist_mean = np.zeros((self.n_dim,))
		dist_stddev = np.ones((self.n_dim,)) # 2.0 * 0.5, 2 is range for (-1, 1), 0.5 is "Const_{rangeToSigmaFactor}"
		best_fitness_so_far = np.NINF
		if self.dist_stddev_reduce_rate is None:
			self.dist_stddev_reduce_rate = (0.001) ** (1 / n_generation)

		for gen in range(n_generation):
			elites_gene = np.full((self.elite_size, self.n_dim), np.NaN)
			elites_fitness = np.full((self.elite_size,), np.NaN)
			for i in range(n_child):
				newbone_gene = np.random.normal(dist_mean, dist_stddev, (self.n_dim))
				newbone_fitness = self.evaluator(newbone_gene)
				# print("newbone:", newbone_gene, newbone_fitness)

				if np.count_nonzero(~np.isnan(elites_fitness)) < self.elite_size:
					elites_gene[-1, :] = newbone_gene
					elites_fitness[-1] = newbone_fitness
				else:
					if elites_fitness[-1] < newbone_fitness:
						elites_gene[-1, :] = newbone_gene
						elites_fitness[-1] = newbone_fitness

				indices = np.argsort(-elites_fitness)
				elites_gene = elites_gene[indices]
				elites_fitness = elites_fitness[indices]

				# print("elite gene:", elites_gene)
				# print("elite fitness:", elites_fitness)
			best_fitness_so_far = max(best_fitness_so_far, elites_fitness[0])
			dist_mean += self.dist_mean_move * (np.mean(elites_gene, axis=0) - dist_mean)
			dist_stddev *= self.dist_stddev_reduce_rate

			print(f"{gen}/{n_generation} fitness:{elites_fitness[0]}, {best_fitness_so_far}")
			# print("mean:", dist_mean, ", stddev:", dist_stddev)

		return best_fitness_so_far

if __name__ == "__main__":
	def sphere(x):
		return -np.sum(x ** 2)

	def sphere_offset(x):
		return -np.sum((x - 0.5) ** 2)

	shclbnd = SHCLVND(2, sphere)
	fitness = shclbnd.train(n_generation = 300, n_child = 5)
	print(fitness)
