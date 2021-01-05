import datetime
# import plot
import numpy as np

class Individual:
	def __init__(self, n):
		# initialized by random values
		if isinstance(n, int):
			self.n = n
			self.gene = np.random.uniform(0, 1, n)
		# initialized by list
		else:
			self.n = len(n)
			self.gene = n
		self.fitness = None
		self.last_gen_mean_fitness = None
	def __str__(self):
		return "{f}:{g}".format(
			f = 'None' if self.fitness is None else '{:.4}'.format(self.fitness),
			g = ','.join(['{:.4}'.format(g) for g in self.gene]))

class JGG:
	def __init__(self, n, npop, npar, nchi, problem):
		self.n = n
		self.npar = npar
		self.nchi = nchi
		self.eval_count = 0
		self.problem = problem
		self.population = [Individual(self.n) for i in range(npop)]
		for i in self.population:
			i.fitness = self.problem(i.gene)
		self.history = {}
		self.history[0] = self.get_best_fitness()

	def select_for_reproduction(self):
		np.random.shuffle(self.population)
		parents = self.population[:self.npar]
		self.population = self.population[self.npar:]
		return parents

	def crossover(self, parents):
		mu = len(parents)
		mean = np.mean(np.array([parent.gene for parent in parents]), axis = 0)
		children = [Individual(self.n) for i in range(self.nchi)]
		for child in children:
			epsilon = np.random.uniform(-np.sqrt(3 / mu), np.sqrt(3 / mu), mu)
			child.gene = mean + np.sum(
				[epsilon[i] * (parents[i].gene - mean) for i in range(mu)], axis = 0)
		return children

	def select_for_survival(self, children):
		children.sort(key = lambda child: -child.fitness)
		return children[:self.npar]

	def evaluate(self, pop):
		self.last_gen_mean_fitness = 0.0
		for individual in pop:
			individual.fitness = self.problem(individual.gene)
			self.last_gen_mean_fitness += individual.fitness / len(pop)
		self.eval_count += len(pop)
		return pop

	def alternation(self):
		parents = self.select_for_reproduction()
		children = self.crossover(parents)
		self.evaluate(children)
		elites = self.select_for_survival(children)
		self.population.extend(elites)
		self.history[self.eval_count] = self.get_best_fitness()

	def until(self, goal, max_eval_count):
		while self.eval_count < max_eval_count:
			self.alternation()
			if self.get_best_fitness() < goal:
				return True
		return False

	def get_best_fitness(self):
		self.population.sort(key = lambda s: -s.fitness if -s.fitness else -np.inf)
		return self.population[0].fitness

	def get_eval_count(self):
		return len(self.history) * self.nchi

# if __name__ == '__main__':
# 	n = 20
# 	ga = JGG(n, 6 * n, n + 1, 6 * n, lambda x: np.sum((x * 10.24 - 5.12) ** 2))

# 	while ga.eval_count < 30000:
# 		ga.alternation()

# 	filename = "benchmark/{0:%Y-%m-%d_%H-%M-%S}.csv".format(datetime.datetime.now())
# 	with open(filename, "w") as f:
# 		for c, v in ga.history.items():
# 			f.write("{0},{1}\n".format(c, v))
# 		f.close()

# 	# plot.plot(filename)

if __name__ == '__main__':
	def sphere(x):
		return -np.sum(x ** 2)

	def sphere_offset(x):
		return -np.sum((x - 0.5) ** 2)

	n_dim = 20
	n_epoch = 100
	batch_size = 100
	# evaluator = sphere
	# y = np.array([0.0 for _ in range(n_dim)])
	evaluator = sphere_offset

	jgg_ = JGG(
		n = n_dim,
		npop = 6 * n_dim,
		npar = n_dim + 1,
		nchi = 6 * n_dim,
		problem = evaluator)
	print(n_epoch * batch_size // 2)
	while jgg_.eval_count < n_epoch * batch_size // 2:
		jgg_.alternation()
		print(jgg_.last_gen_mean_fitness, jgg_.get_best_fitness())
	# jgg_.get_best_fitness()
