import os
import numpy as np
import swapgan
import cheatynn
import greedynn
import greedynn_rand
import greedynn_mp
import greedynn_mp_rand
import greedynn_mp_mem
import greedynn_mp_rand_mem
import randgen
import jgg
import cmaes
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
n_loop = 10
# evaluator = sphere          ; n_gen_img = 100; n_epoch = 100
# evaluator = sphere          ; n_gen_img = 100; n_epoch = 1000
# evaluator = sphere_offset   ; n_gen_img = 100; n_epoch = 100
evaluator = sphere_offset   ; n_gen_img = 100; n_epoch = 1000
# evaluator = ackley          ; n_gen_img = 100; n_epoch = 1000
# evaluator = ackley          ; n_gen_img = 100; n_epoch = 10000
# evaluator = ackley_offset   ; n_gen_img = 100; n_epoch = 1000
# evaluator = ackley_offset   ; n_gen_img = 100; n_epoch = 10000
# evaluator = rastrigin       ; n_gen_img = 100; n_epoch = 1000
# evaluator = rastrigin       ; n_gen_img = 100; n_epoch = 10000
# evaluator = rastrigin_offset; n_gen_img = 100; n_epoch = 1000
# evaluator = rastrigin_offset; n_gen_img = 100; n_epoch = 10000

batch_size = 10
# y = np.array([0.0 for _ in range(n_dim)])
# y = np.array([0.5 for _ in range(n_dim)])

env_str = f"function={evaluator.__name__}({n_dim}dim),n_gen_img={n_gen_img},n_epoch={n_epoch},batch_size={batch_size},n_loop={n_loop}"
bench_dir = f"benchmark/{env_str}/"
os.makedirs(bench_dir, exist_ok=True)

for loop in range(n_loop):
	# greedynn_ = greedynn.GreedyNN(
	# 	img_shape = (1, n_dim),
	# 	n_gen_img = n_gen_img,
	# 	evaluator = evaluator,
	# 	noise_dim = 1,
	# 	filepath = f"{bench_dir}greedynn_{loop}.csv")
	# f = greedynn_.train(n_epoch=n_epoch, batch_size=batch_size)

	# greedynn_rand_ = greedynn_rand.GreedyNN_RAND(
	# 	img_shape = (1, n_dim),
	# 	n_gen_img = n_gen_img,
	# 	evaluator = evaluator,
	# 	noise_dim = 1,
	# 	filepath = f"{bench_dir}greedynn_rand_{loop}.csv")
	# f = greedynn_rand_.train(n_epoch=n_epoch, batch_size=batch_size)

	lr = 0.015
	greedynn_mp_ = greedynn_mp.GreedyNN_MP(
		img_shape = (10, n_dim),
		n_gen_img = n_gen_img,
		evaluator = evaluator,
		lr = lr,
		noise_dim = 1,
		filepath = f"{bench_dir}greedynn_mp_lr{lr}_{loop}.csv")
	f = greedynn_mp_.train(n_epoch=n_epoch, batch_size=batch_size)

	lr = 0.015
	greedynn_mp_rand_ = greedynn_mp_rand.GreedyNN_MP_RAND(
		img_shape = (10, n_dim),
		n_gen_img = n_gen_img,
		evaluator = evaluator,
		lr = lr,
		noise_dim = 1,
		filepath = f"{bench_dir}greedynn_mp_rand_lr{lr}_{loop}.csv")
	f = greedynn_mp_rand_.train(n_epoch=n_epoch, batch_size=batch_size)

	lr = 0.015
	greedynn_mp_mem_ = greedynn_mp_mem.GreedyNN_MP_MEM(
		img_shape = (10, n_dim),
		n_gen_img = n_gen_img,
		evaluator = evaluator,
		lr = lr,
		noise_dim = 1,
		filepath = f"{bench_dir}greedynn_mp_mem_lr{lr}_{loop}.csv")
	f = greedynn_mp_mem_.train(n_epoch=n_epoch, batch_size=batch_size)

	lr = 0.015
	greedynn_mp_rand_mem_ = greedynn_mp_rand_mem.GreedyNN_MP_RAND_MEM(
		img_shape = (10, n_dim),
		n_gen_img = n_gen_img,
		evaluator = evaluator,
		lr = lr,
		noise_dim = 1,
		filepath = f"{bench_dir}greedynn_mp_rand_mem_lr{lr}_{loop}.csv")
	f = greedynn_mp_rand_mem_.train(n_epoch=n_epoch, batch_size=batch_size)

	# cmaes_ = cmaes.CMAES(
	# 	n_dim = n_dim,
	# 	evaluator = evaluator)
	# f = cmaes_.train(max_eval_count = n_epoch * batch_size // 2)

	# pbilc_ = pbilc.PBILC(
	# 	n_dim = n_dim,
	# 	n_child = n_gen_img,
	# 	evaluator = evaluator,
	# 	filepath = f"{bench_dir}pbilc_{loop}.csv")
	# f = pbilc_.train(n_generation = n_epoch)

	f = pso.particleswarm(
		evaluator = lambda x: -evaluator(np.array(x, dtype=np.float)),
		bounds = [[-1, 1] for _ in range(n_dim)],
		max_n_eval = n_epoch * n_gen_img,
		filepath = f"{bench_dir}pso_{loop}.csv")

print(env_str)
