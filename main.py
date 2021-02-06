import os
import numpy as np
import greedynn
import greedynn_mp
import pso

def sphere(x):
	return -np.sum(x ** 2)

def sphere_offset(x):
	return -np.sum((x - 0.5) ** 2)

def ackley(x_):
	x = np.copy(x_) * 32.768
	return -20 + 20 * np.exp(- 0.2 * np.sqrt(1.0 / len(x) * np.sum(x ** 2)))\
		- np.e + np.exp(1.0 / len(x) * np.sum(np.cos(2 * np.pi * x)))

def ackley_offset(x_):
	x = (np.copy(x_) - 0.5) * 32.768
	return -20 + 20 * np.exp(- 0.2 * np.sqrt(1.0 / len(x) * np.sum(x ** 2)))\
		- np.e + np.exp(1.0 / len(x) * np.sum(np.cos(2 * np.pi * x)))

def rastrigin(x_):
	x = np.copy(x_) * 5.12
	return -10 * len(x) - np.sum(x ** 2) + 10 * np.sum(np.cos(2 * np.pi * x))

def rastrigin_offset(x_):
	x = (np.copy(x_) - 0.5) * 5.12
	return -10 * len(x) - np.sum(x ** 2) + 10 * np.sum(np.cos(2 * np.pi * x))

n_dim = 20
n_loop = 1
evaluator = sphere          ; optimum = [0.0] * n_dim; n_eval = int(1e2)
# evaluator = sphere_offset   ; optimum = [0.5] * n_dim; n_eval = int(1e5)
# evaluator = ackley          ; optimum = [0.0] * n_dim; n_eval = int(1e6)
# evaluator = ackley_offset   ; optimum = [0.5] * n_dim; n_eval = int(1e6)
# evaluator = rastrigin       ; optimum = [0.0] * n_dim; n_eval = int(1e6)
# evaluator = rastrigin_offset; optimum = [0.5] * n_dim; n_eval = int(1e6)

# y = np.array([0.0 for _ in range(n_dim)])
# y = np.array([0.5 for _ in range(n_dim)])

env_str = f"fffffunction={evaluator.__name__}({n_dim}dim),n_eval={n_eval},n_loop={n_loop}"
bench_dir = f"benchmark/{env_str}/"
os.makedirs(bench_dir, exist_ok=True)

for loop in range(n_loop):
	lr = 0.015
	p = 1
	greedynn_ = greedynn.GreedyNN(
		img_shape = (p, n_dim),
		evaluator = evaluator,
		optimum = optimum,
		lr = lr,
		noise_dim = 1,
		filepath = f"{bench_dir}greedynn_p{p}_{loop}.csv")
	f = greedynn_.train(max_n_eval = n_eval, n_batch = 10, batch_size = 10)

	lr = 0.015
	p = 3
	greedynn_mp_ = greedynn_mp.GreedyNN_MP(
		img_shape = (p, n_dim),
		evaluator = evaluator,
		optimum = optimum,
		lr = lr,
		noise_dim = 1,
		filepath = f"{bench_dir}greedynn_p{p}_{loop}.csv")
	f = greedynn_mp_.train(max_n_eval = n_eval, n_batch = 10, batch_size = 15)

	lr = 0.015
	p = 8
	greedynn_mp_ = greedynn_mp.GreedyNN_MP(
		img_shape = (p, n_dim),
		evaluator = evaluator,
		optimum = optimum,
		lr = lr,
		noise_dim = 1,
		filepath = f"{bench_dir}greedynn_p{p}_{loop}.csv")
	f = greedynn_mp_.train(max_n_eval = n_eval, n_batch = 10, batch_size = 16)

	n_particles = n_dim * 100
	f = pso.pso(
		evaluator = lambda x: -evaluator(np.array(x, dtype=np.float)),
		optimum = optimum,
		n_dim = n_dim,
		n_particles = n_particles,
		max_n_eval = n_eval,
		filepath = f"{bench_dir}pso_p{n_particles}_{loop}.csv")

print(env_str)
