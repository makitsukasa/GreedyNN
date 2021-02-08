import os
import numpy as np
import greedynn
import greedynn_mp
import greedynn_mp_greedy
import greedynn_vp
import pso
from problem.frontiers import *

n_dim = 20
n_loop = 10

# setting = (sphere_offset, [0.5] * n_dim, int(1e4), 3, 15, n_dim * 10) # -0.1
# setting = (ktablet_offset, [0.5] * n_dim, int(5e4), 3, 15, n_dim * 10) # -2000
# setting = (bohachevsky_offset, [0.5] * n_dim, int(3e4), 5, 15, n_dim * 10) # -10
# setting = (ackley_offset, [0.5] * n_dim, int(7e4), 5, 15, n_dim * 100) # -1
# setting = (ackley_offset, [0.5] * n_dim, int(5e6), 10, 20, n_dim * 100) #
# setting = (schaffer_offset, [0.5] * n_dim, int(1e6), 5, 15, n_dim * 100) #
# setting = (rastrigin_offset, [0.5] * n_dim, int(5e5), 10, 20, n_dim * 100) # -20
# setting = (rastrigin_offset, [0.5] * n_dim, int(5e6+1), 40, 160, n_dim * 100) #
# evaluator, optimum, n_eval, p, batch_size, n_particles = setting

# evaluator, n_eval = sphere_offset, int(5e4 + 1)
# evaluator, n_eval = ktablet_offset, int(1e5)
# evaluator, n_eval = bohachevsky_offset, int(1e5)
# evaluator, n_eval = ackley_offset, int(1e5)
# evaluator, n_eval = schaffer_offset, int(5e5)
evaluator, n_eval = rastrigin_offset, int(5e5)
optimum, p, batch_size, n_particles = [0.5] * n_dim, 5, 15, n_dim * 100

# y = np.array([0.0 for _ in range(n_dim)])
# y = np.array([0.5 for _ in range(n_dim)])

env_str = f"function={evaluator.__name__}({n_dim}dim),n_eval={n_eval},n_loop={n_loop}"
bench_dir = f"benchmark/{env_str}/"
os.makedirs(bench_dir, exist_ok=True)

for loop in range(n_loop):
	lr = 0.01
	n = greedynn_mp.GreedyNN_MP(
		img_shape = (p, n_dim),
		evaluator = evaluator,
		optimum = optimum,
		lr = lr,
		noise_dim = 3,
		filepath = f"{bench_dir}提案解法_{loop}.csv")
	f = n.train(max_n_eval = n_eval, n_batch = 10, batch_size = batch_size)

	# lr = 0.01
	# n = greedynn_mp_greedy.GreedyNN_MP_Greedy(
	# 	img_shape = (p, n_dim),
	# 	evaluator = evaluator,
	# 	optimum = optimum,
	# 	lr = lr,
	# 	noise_dim = 3,
	# 	filepath = f"{bench_dir}貪欲_{loop}.csv")
	# f = n.train(max_n_eval = n_eval, n_batch = 10, batch_size = batch_size)

	# lr = 0.01
	# n = greedynn_vp.GreedyNN_VP(
	# 	img_shape = (p, n_dim),
	# 	evaluator = evaluator,
	# 	optimum = optimum,
	# 	lr = lr,
	# 	noise_dim = 3,
	# 	filepath = f"{bench_dir}greedynn_vp_p{p}_{loop}.csv")
	# f = n.train(max_n_eval = n_eval, n_batch = 10, batch_size = batch_size)

	f = pso.pso(
		evaluator = lambda x: -evaluator(np.array(x, dtype=np.float)),
		optimum = optimum,
		n_dim = n_dim,
		n_particles = n_particles,
		max_n_eval = n_eval,
		filepath = f"{bench_dir}PSO_{loop}.csv")

print(env_str)
