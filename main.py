import os
import numpy as np
import swapgan
import cheatynn
import greedynn
import greedynn_multipoint
import greedynn_mp_lstsq
import greedynn_mp_polyfit
import randgen
import jgg
import cmaes
import pbilc

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
evaluator = sphere          ; n_gen_img = 100; n_epoch = 100
# evaluator = sphere          ; n_gen_img = 100; n_epoch = 1000
# evaluator = sphere_offset   ; n_gen_img = 100; n_epoch = 100
# evaluator = sphere_offset   ; n_gen_img = 100; n_epoch = 1000
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

f_swapgan = [None for _ in range(n_loop)]
f_fswapgan = [None for _ in range(n_loop)]
f_cheatynn = [None for _ in range(n_loop)]
f_greedynn = [None for _ in range(n_loop)]
f_greedynn2 = [None for _ in range(n_loop)]
f_fgreedynn = [None for _ in range(n_loop)]
f_fgreedynn2 = [None for _ in range(n_loop)]
f_greedynn_mp = [None for _ in range(n_loop)]
f_greedynn_mp_lst = [None for _ in range(n_loop)]
f_greedynn_mp_pf = [None for _ in range(n_loop)]
f_randgen = [None for _ in range(n_loop)]
f_jgg = [None for _ in range(n_loop)]
f_cmaes = [None for _ in range(n_loop)]
f_pbilc = [None for _ in range(n_loop)]

env_str = f"function={evaluator.__name__}({n_dim}dim),n_gen_img={n_gen_img},n_epoch={n_epoch},batch_size={batch_size},n_loop={n_loop}"
bench_dir = f"benchmark/{env_str}/"
os.makedirs(bench_dir, exist_ok=True)

for loop in range(n_loop):
	pass
	# swapgan_ = swapgan.SwapGAN(
	# 	img_shape = (n_dim, 1),
	# 	train_img = np.random.uniform(-1, 1, (n_gen_img, n_dim, 1)),
	# 	evaluator = evaluator,
	# 	z_dim = 1)
	# f = swapgan_.train(n_epoch=n_epoch, batch_size=batch_size)
	# f_swapgan[loop] = f

	# fswapgan_ = swapgan.SwapGAN(
	# 	img_shape = (n_dim, 1),
	# 	train_img = np.random.uniform(-1, 1, (n_gen_img, n_dim, 1)),
	# 	evaluator = evaluator,
	# 	z_dim = 1,
	# 	fixed_noise=True)
	# f = fswapgan_.train(n_epoch=n_epoch, batch_size=batch_size)
	# f_fswapgan[loop] = f

	# greedynn_ = greedynn.GreedyNN(
	# 	img_shape = (1, n_dim),
	# 	n_gen_img = n_gen_img,
	# 	evaluator = evaluator,
	# 	noise_dim = 1)
	# f = greedynn_.train(n_epoch=n_epoch, batch_size=batch_size//2)
	# f_greedynn[loop] = f

	# greedynn_2 = greedynn.GreedyNN(
	# 	img_shape = (1, n_dim),
	# 	n_gen_img = n_gen_img // 10,
	# 	evaluator = evaluator,
	# 	noise_dim = 1)
	# f = greedynn_2.train(n_epoch=n_epoch * 10, batch_size=batch_size//2)
	# f_greedynn2[loop] = f

	# fgreedynn_ = greedynn.GreedyNN(
	# 	img_shape = (1, n_dim),
	# 	n_gen_img = n_gen_img,
	# 	evaluator = evaluator,
	# 	noise_dim = 1,
	# 	fixed_noise=True)
	# f = fgreedynn_.train(n_epoch=n_epoch, batch_size=batch_size//2)
	# f_fgreedynn[loop] = f

	# fgreedynn_2 = greedynn.GreedyNN(
	# 	img_shape = (1, n_dim),
	# 	n_gen_img = n_gen_img // 10,
	# 	evaluator = evaluator,
	# 	noise_dim = 1,
	# 	fixed_noise=True)
	# f = fgreedynn_2.train(n_epoch=n_epoch * 10, batch_size=batch_size//2)
	# f_fgreedynn2[loop] = f

	# jgg_ = jgg.JGG(
	# 	n = n_dim,
	# 	npop = 6 * n_dim,
	# 	npar = n_dim + 1,
	# 	nchi = 6 * n_dim,
	# 	problem = evaluator)
	# while jgg_.eval_count < n_epoch * batch_size//2:
	# 	jgg_.alternation()
	# 	print(jgg_.last_gen_mean_fitness, jgg_.get_best_fitness())
	# f_jgg[loop] = jgg_.get_best_fitness()

	# cmaes_ = cmaes.CMAES(
	# 	n_dim = n_dim,
	# 	evaluator = evaluator)
	# f = cmaes_.train(max_eval_count = n_epoch * batch_size // 2)
	# f_cmaes[loop] = f

for loop in range(n_loop):
	greedynn_ = greedynn.GreedyNN(
		img_shape = (1, n_dim),
		n_gen_img = n_gen_img,
		evaluator = evaluator,
		noise_dim = 1,
		filepath = f"{bench_dir}greedynn_{loop}.csv")
	f = greedynn_.train(n_epoch=n_epoch, batch_size=batch_size)
	f_greedynn[loop] = f

	greedynn_mp_pf_ = greedynn_mp_polyfit.GreedyNN_MP_PF(
		img_shape = (5, n_dim),
		n_gen_img = n_gen_img,
		evaluator = evaluator,
		noise_dim = 1,
		filepath = f"{bench_dir}greedynn_mp_pf_{loop}.csv")
	f = greedynn_mp_pf_.train(n_epoch=n_epoch, batch_size=batch_size)
	f_greedynn_mp_pf[loop] = f

	# cmaes_ = cmaes.CMAES(
	# 	n_dim = n_dim,
	# 	evaluator = evaluator)
	# f = cmaes_.train(max_eval_count = n_epoch * batch_size // 2)
	# f_cmaes[loop] = f

	pbilc_ = pbilc.PBILC(
		n_dim = n_dim,
		n_child = n_gen_img,
		evaluator = evaluator,
		filepath = f"{bench_dir}pbilc_{loop}.csv")
	f = pbilc_.train(n_generation = n_epoch)
	f_pbilc[loop] = f

print(env_str)
# print("swap   :", np.mean(f_swapgan))
# print("fswap  :", np.mean(f_fswapgan))
print("greed  :", np.mean(f_greedynn))
# print("greedy2:", np.mean(f_greedynn2))
# print("fgreedy:", np.mean(f_fgreedynn))
# print("fgreed2:", np.mean(f_fgreedynn2))
# print("greedmp:", np.mean(f_greedynn_mp))
# print("grmplst:", np.mean(f_greedynn_mp_lst))
print("grmppf:", np.mean(f_greedynn_mp_pf))
# print("jgg    :", np.mean(f_jgg))
# print("cmaes  :", np.mean(f_cmaes))
print("eda    :", np.mean(f_pbilc))
