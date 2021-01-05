# https://gist.github.com/yukoba/731bb555a7dc79cdfecf33904ee4e043

#############################################################
# 最適化アルゴリズム CMA-ES の実装。
# https://www.lri.fr/~hansen/purecmaes.m を Python 3 に移植。
# https://arxiv.org/abs/1604.00772 も参考にしています。
#############################################################

import numpy as np

class CMAES:
	def __init__(self, n_dim, evaluator, step_size=1.0, filepath = None):
		self.n_dim = n_dim
		self.step_size = step_size
		self.evaluator = evaluator
		self.filepath = filepath

		# Strategy parameter setting: Selection
		self.n_pop = 4 + int(3 * np.log(self.n_dim))
		# self.n_pop *= 5  # 最小値に到達する確率を上げるために、独自に増やしました
		self.n_par = int(self.n_pop / 2)
		self.weights = np.log((self.n_pop + 1) / 2) - np.log(np.arange(self.n_par) + 1) # muXone recombination weights
		self.weights /= self.weights.sum()
		self.mass = (self.weights.sum() ** 2) / (self.weights ** 2).sum() # \mu_{eff} : variance effective selection mass

		# Strategy parameter setting: Adaptation
		self.cc = (4 + self.mass / self.n_dim) / (self.n_dim + 4 + 2 * self.mass / self.n_dim)  # (56)
		self.cs = (self.mass + 2) / (self.n_dim + self.mass + 5)
		alpha_cov = 2
		self.c1 = alpha_cov / ((self.n_dim + 1.3) ** 2 + self.mass)  # (57)
		self.cmu = min(1 - self.c1, alpha_cov * (self.mass - 2 + 1 / self.mass) / ((self.n_dim + 2) ** 2 + alpha_cov * self.mass / 2))  # (58)
		self.damps = 1 + 2 * max(0, np.sqrt((self.mass - 1) / (self.n_dim + 1)) - 1) + self.cs  # (55)

	def train(self, goal_fitness=1e-10, max_eval_count=None):
		self.max_eval_count = 1e3 * self.n_dim ** 2 if max_eval_count is None else max_eval_count
		self.xmean = np.random.randn(self.n_dim, 1)

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

		# Initialize dynamic (internal) strategy parameters and constants
		pc = np.zeros([self.n_dim, 1])
		ps = np.zeros([self.n_dim, 1])
		B = np.eye(self.n_dim)
		D = np.ones([self.n_dim, 1])
		C = B @ np.diag((D ** 2).flatten()) @ B.T
		invsqrtC = B @ np.diag((D ** -1).flatten()) @ B.T
		eigenval = 0
		chiN = self.n_dim ** 0.5 * (1 - 1 / (4 * self.n_dim) + 1 / (21 + self.n_dim ** 2))

		eval_count = 0
		while eval_count < self.max_eval_count:
			# Generate and evaluate lambda offspring
			arx = self.xmean + self.step_size * (B @ (D * np.random.randn(self.n_dim, self.n_pop)))
			# arfitness = self.evaluator(arx)
			arfitness = np.array([self.evaluator(np.array(x)) for x in arx.T.tolist()])
			eval_count += self.n_pop

			# Sort by fitness and compute weighted mean into xmean
			arindex = np.argsort(-arfitness)
			arfitness = arfitness[arindex]
			xold = self.xmean
			self.xmean = (arx[:, arindex[:self.n_par]] @ self.weights).reshape([-1, 1])

			# Cumulation: Update evolution paths
			ps = (1 - self.cs) * ps + np.sqrt(self.cs * (2 - self.cs) * self.mass) * invsqrtC @ (self.xmean - xold) / self.step_size
			hsig = (ps ** 2).sum() / (1 - (1 - self.cs) ** (2 * eval_count / self.n_pop)) / self.n_dim < 2 + 4 / (self.n_dim + 1)
			pc = (1 - self.cc) * pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mass) * (self.xmean - xold) / self.step_size

			# Adapt covariance matrix C
			artmp = (1 / self.step_size) * (arx[:, arindex[:self.n_par]] - xold)
			C = (1 - self.c1 - self.cmu) * C + \
				self.c1 * (pc @ pc.T + (1 - hsig) * self.cc * (2 - self.cc) * C) + \
				self.cmu * artmp @ np.diag(self.weights) @ artmp.T

			# Adapt step size
			self.step_size *= np.exp((self.cs / self.damps) * (np.linalg.norm(ps) / chiN - 1))

			# Update B and D from C
			if eval_count - eigenval > self.n_pop / (self.c1 + self.cmu) / self.n_dim / 10:
				eigenval = eval_count
				# C = np.triu(C) + np.triu(C, 1).T
				D, B = np.linalg.eigh(C, 'U')
				D = np.sqrt(D.real).reshape([self.n_dim, 1])
				invsqrtC = B @ np.diag((D ** -1).flatten()) @ B.T

			# if arfitness[0] <= goal_fitness or D.max() > 1e7 * D.min():
			# 	break

			if np.allclose(arfitness[0], arfitness[int(1 + 0.7 * self.n_pop)], 1e-10, 1e-10):
				self.step_size *= np.exp(0.2 + self.cs / self.damps)

			print("%d:%g,%g" % (eval_count, np.mean(arfitness), arfitness[0]))

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

		return arfitness[0]

if __name__ == "__main__":
	def sphere(x):
		return -np.sum(x ** 2)

	def sphere_offset(x):
		return -np.sum((x - 0.5) ** 2)

	n_dim = 10
	n_loop = 1
	n_gen_img = 1000
	n_epoch = 20
	batch_size = 100
	# evaluator = sphere
	evaluator = sphere_offset

	cmaes = CMAES(n_dim = n_dim, evaluator = evaluator)
	f = cmaes.train(max_eval_count = n_epoch * batch_size // 2)
	# f = cmaes.train(max_eval_count = 500)
	print(f)
