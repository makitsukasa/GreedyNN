import os
import numpy as np
import greedynn
import greedynn_mp
import greedynn_mp_fixed
import greedynn_mp_greedy
import greedynn_mp_notrain
import greedynn_mp_distribution
import greedynn_vpi
import greedynn_vpd
import pso
from problem.frontiers import *

n_dim = 20
n_loop = 10

evaluator, n_eval = sphere_offset, int(5e4)
# evaluator, n_eval = ktablet_offset, int(1e5)
# evaluator, n_eval = bohachevsky_offset, int(1e5)
# evaluator, n_eval = ackley_offset, int(1e5)
# evaluator, n_eval = schaffer_offset, int(5e5)
# evaluator, n_eval = rastrigin_offset, int(5e5)
optimum = [0.5] * n_dim; p = 5; batch_size = 15; n_particles = n_dim * 100
# n_eval = 75 * 100

# y = np.array([0.0 for _ in range(n_dim)])
# y = np.array([0.5 for _ in range(n_dim)])

env_str = f"function={evaluator.__name__}({n_dim}dim),n_eval={n_eval},n_loop={n_loop}"
bench_dir = f"benchmark/{env_str}/"
os.makedirs(bench_dir, exist_ok=True)

for loop in range(n_loop):
	# lr = 0.01
	# n = greedynn.GreedyNN(
	# 	img_shape = (1, n_dim),
	# 	evaluator = evaluator,
	# 	optimum = optimum,
	# 	lr = lr,
	# 	noise_dim = 3,
	# 	filepath = f"{bench_dir}1点_{loop}.csv")
	# f = n.train(max_n_eval = n_eval, n_batch = 10, batch_size = batch_size)

	lr = 0.01
	n = greedynn_mp.GreedyNN_MP(
		img_shape = (p, n_dim),
		evaluator = evaluator,
		optimum = optimum,
		lr = lr,
		noise_dim = 3,
		filepath = f"{bench_dir}提案法_{loop}.csv")
	f = n.train(max_n_eval = n_eval, n_batch = 10, batch_size = batch_size)

	lr = 0.01
	n = greedynn_mp_fixed.GreedyNN_MP_FIXED(
		img_shape = (p, n_dim),
		evaluator = evaluator,
		optimum = optimum,
		lr = lr,
		noise_dim = 3,
		filepath = f"{bench_dir}goodが最適解_{loop}.csv")
	f = n.train(train_data = np.array([[[0.5] * n_dim] * p] * batch_size),
		max_n_eval = n_eval, n_batch = 10, batch_size = batch_size)

	lr = 0.01
	n = greedynn_mp_fixed.GreedyNN_MP_FIXED(
		img_shape = (p, n_dim),
		evaluator = evaluator,
		optimum = optimum,
		lr = lr,
		noise_dim = 3,
		filepath = f"{bench_dir}goodが遠い_{loop}.csv")
	f = n.train(train_data = np.array([[0.5] * n_dim + [0.0] * n_dim + [0.0] * n_dim +
		[1.0] * n_dim + [1.0] * n_dim] * batch_size).reshape(batch_size, p, n_dim),
		max_n_eval = n_eval, n_batch = 10, batch_size = batch_size)

	# f = pso.pso(
	# 	evaluator = lambda x: -evaluator(np.array(x, dtype=np.float)),
	# 	optimum = optimum,
	# 	n_dim = n_dim,
	# 	n_particles = n_particles,
	# 	max_n_eval = n_eval,
	# 	filepath = f"{bench_dir}PSO_{loop}.csv")

	# lr = 0.01
	# n = greedynn_mp_distribution.GreedyNN_MP_distribution(
	# 	img_shape = (p, n_dim),
	# 	evaluator = evaluator,
	# 	optimum = optimum,
	# 	lr = lr,
	# 	noise_dim = 3,
	# 	filepath = f"{bench_dir}提案解法_{loop}.csv",
	# 	filepath_distribution_before=f"benchmark/distribution/{evaluator.__name__}_before.csv",
	# 	filepath_distribution_after=f"benchmark/distribution/{evaluator.__name__}_after.csv")
	# f = n.train(max_n_eval = n_eval, n_batch = 10, batch_size = batch_size)

print(env_str)
