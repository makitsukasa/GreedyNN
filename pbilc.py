# PBILC
# algorithm and parameters are in https://mnakao.net/data/2009/EC-paper.pdf

import csv
import numpy as np

class PBILC:
	def __init__(self, n_dim, n_child, evaluator, learning_rate = 0.005, elite_rate = 0.5, filepath = None):
		self.n_dim = n_dim
		self.evaluator = evaluator
		self.n_child = n_child
		self.learning_rate = learning_rate
		self.elite_rate = elite_rate
		self.filepath = filepath

	def train(self, n_generation):
		dist_mean = np.zeros((self.n_dim,))
		dist_stddev = np.ones((self.n_dim,)) # 2.0 * 0.5, 2 is range for (-1, 1), 0.5 is "Const_{rangeToSigmaFactor}"
		best_fitness_so_far = np.NINF

		if self.filepath:
			f = open(self.filepath, mode = "w")
			csv_writer = csv.writer(f)
			csv_writer.writerow([
				"n_eval",
				"max_n_eval",
				"dist_mean",
				"dist_stddev",
				"fitness_mean",
				"fitness_best",
				"fitness_best_so_far",
			])

		for gen in range(n_generation):
			children_gene = np.full((self.n_child, self.n_dim), np.NaN)
			children_fitness = np.full((self.n_child,), np.NaN)
			for i in range(self.n_child):
				children_gene[i, :] = np.random.normal(dist_mean, dist_stddev, (self.n_dim))
				children_fitness[i] = self.evaluator(children_gene[i, :])

			indices = np.argsort(-children_fitness)
			children_gene = children_gene[indices]
			children_fitness = children_fitness[indices]

			# µ ← (1−α)µ+α(Xbest1+Xbest2−Xworst)
			dist_mean += -self.learning_rate * dist_mean + self.learning_rate * (children_gene[0] + children_gene[1] - children_gene[-1])
			# σ ← (1−α)σ+α*stddev(elites in X)
			elites_gene = children_gene[:int(len(children_fitness) * self.elite_rate)]
			dist_stddev += -self.learning_rate * dist_stddev + self.learning_rate * np.std(elites_gene, axis=0)

			best_fitness_so_far = max(best_fitness_so_far, children_fitness[0])

			print(f"{(gen + 1) * self.n_child}/{n_generation * self.n_child} fitness:{children_fitness[0]}, {best_fitness_so_far}")
			# print(children_gene[0])
			print("mean:", np.mean(dist_mean), ", stddev:", np.mean(dist_stddev))

			if self.filepath:
				csv_writer.writerow([
					(gen + 1) * self.n_child,
					n_generation * self.n_child,
					np.mean(np.mean(children_gene, axis=0)),
					np.mean(np.std(children_gene, axis=0)),
					np.mean(children_fitness),
					children_fitness[0],
					best_fitness_so_far,
				])

		if self.filepath:
			f.close()

		return best_fitness_so_far

if __name__ == "__main__":
	def sphere(x):
		return -np.sum(x ** 2)

	def sphere_offset(x):
		return -np.sum((x - 0.5) ** 2)

	def ackley(x):
		x *= 32.768
		return -20 + 20 * np.exp(- 0.2 * np.sqrt(1.0 / len(x) * np.sum(x ** 2)))\
			- np.e + np.exp(1.0 / len(x) * np.sum(np.cos(2 * np.pi * x)))

	def rastrigin(x):
		x *= 5.12
		return -10 * len(x) - np.sum(x ** 2) + 10 * np.sum(np.cos(2 * np.pi * x))

	shclbnd = PBILC(
		n_dim = 5,
		n_child = 20,
		evaluator = rastrigin,
		learning_rate = 0.005)
	fitness = shclbnd.train(n_generation = 100)
	print(fitness)
