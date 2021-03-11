### -*-coding:utf-8-*-
import os
import sys
import csv
import numpy as np

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomUniform
import warnings

warnings.simplefilter('ignore', np.RankWarning)
np.set_printoptions(formatter={'float': '{:.3}'.format})

class GreedyNN_MP_distribution():
	def __init__(
			self,
			img_shape,
			evaluator,
			optimum,
			lr = 0.01,
			noise_dim = 100,
			fixed_noise = False,
			filepath = None,
			filepath_distribution_before = None,
			filepath_distribution_after = None):
		self.img_shape = img_shape
		self.noise_dim = noise_dim
		self.evaluator = evaluator
		self.optimum = optimum
		self.fixed_noise = fixed_noise
		self.filepath = filepath
		self.filepath_distribution_before = filepath_distribution_before
		self.filepath_distribution_after = filepath_distribution_after

		optimizer = Adam(lr)

		# Generator model
		self.generator = self.build_generator()
		self.generator.compile(loss='mean_absolute_error', optimizer=optimizer)

	def build_generator(self):
		noise_shape = (self.noise_dim,)
		n_unit = self.img_shape[0] * self.img_shape[1]
		model = Sequential()

		model.add(Dense(n_unit, input_shape=noise_shape))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(n_unit))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(n_unit))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(
			np.prod(n_unit), activation='linear', kernel_initializer=RandomUniform(-1,1)))
		model.add(Reshape(self.img_shape))

		model.summary()

		return model

	def train(self, max_n_eval, n_batch = 10, batch_size = 10):
		print('Number of batches:', n_batch)
		best_fitness = np.NINF
		best_img = np.random.uniform(-1.0, 1.0, (self.img_shape[1]))
		teacher_fitness = np.full((self.img_shape[0] - 1), np.NINF)
		teacher_img = np.full((self.img_shape[0] - 1, self.img_shape[1]), np.NAN)
		noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
		n_eval = 0

		if self.filepath_distribution_before:
			os.makedirs(os.path.dirname(self.filepath_distribution_before), exist_ok=True)
			f = open(self.filepath_distribution_before, mode = "w")
			csv_writer = csv.writer(f)
			for _ in range(10):
				noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
				gen_imgs = self.generator.predict(noise)
				gen_imgs.reshape(-1, gen_imgs.shape[2])
				for row in gen_imgs.reshape(-1, gen_imgs.shape[2]):
					csv_writer.writerow(row)
			f.close()

		if self.filepath:
			f = open(self.filepath, mode = "w")
			csv_writer = csv.writer(f)
			csv_writer.writerow([
				"n_eval",
				"max_n_eval",
				"dist_r",
				"dist_stddev",
				"train_loss",
				"fitness_mean",
				"fitness_best",
				"fitness_best_so_far",
			])

		while n_eval < max_n_eval:
			for iteration in range(n_batch):
				# ---------------------
				#  Generator learning
				# ---------------------
				# pickup images from generator
				if not self.fixed_noise:
					noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
				gen_imgs = self.generator.predict(noise)
				gen_fitness = np.apply_along_axis(self.evaluator, 2, gen_imgs)

				# swap
				ascending_indice = np.unravel_index(
					np.argsort(gen_fitness.flatten()), gen_fitness.shape)
				if gen_fitness[ascending_indice][-1] > best_fitness:
					best_fitness = gen_fitness[ascending_indice][-1]
					best_img = gen_imgs[ascending_indice][-1]

				# Train the generator
				# 近似
				fitness_pred_error = np.copy(gen_fitness)
				teacher_fitness_pred_error = np.copy(teacher_fitness)
				for i in range(gen_imgs.shape[2]):
					p = np.polyfit(gen_imgs[:, :, i].flatten(), gen_fitness.flatten(), 2)
					if p[0] < 0:
						p[0] = p[1] = 0
					y_pred = (p[0] * gen_imgs[:, :, i] ** 2 +
						p[1] * gen_imgs[:, :, i] + p[2]) / gen_imgs.shape[2]
					fitness_pred_error -= np.reshape(y_pred, fitness_pred_error.shape)
					t_pred = (p[0] * teacher_img[:, i] ** 2 +
						p[1] * teacher_img[:, i] + p[2]) / gen_imgs.shape[2]
					teacher_fitness_pred_error -= np.reshape(
						t_pred, teacher_fitness_pred_error.shape)

				vstacked_imgs = np.vstack((gen_imgs.reshape(-1, gen_imgs.shape[2]), teacher_img))
				vstacked_fitnesses = np.hstack((gen_fitness.flatten(), teacher_fitness))

				error_ascending_indice = np.argsort(np.hstack(
					(fitness_pred_error.flatten(), teacher_fitness_pred_error)))
				error_ascending_indice = error_ascending_indice[np.where(
					np.isfinite(vstacked_imgs).all(axis = 1) &
					(vstacked_imgs != best_img).all(axis = 1))]

				teacher_img = vstacked_imgs[error_ascending_indice][-teacher_img.shape[0]:]
				teacher_fitness = vstacked_fitnesses[error_ascending_indice][-teacher_img.shape[0]:]

				gen_error_ascending_indice = np.unravel_index(
					np.argsort(fitness_pred_error.flatten()), fitness_pred_error.shape)

				y_raw = np.append([best_img], teacher_img, axis=0)
				y = np.tile(y_raw, (batch_size, 1, 1))
				for _ in range(1):
					g_loss = self.generator.train_on_batch(noise, y)

				n_eval += batch_size * self.img_shape[0]

				# progress
				print ("eval:%d/%d, iter:%d/%d, [G loss: %f] [mean: %f best: %f]" %
					(n_eval, max_n_eval, iteration+1, n_batch,
					g_loss, np.mean(gen_fitness), best_fitness))
				print(f"b {best_fitness:.3} t {teacher_fitness}")

				r = np.sqrt(np.sum((best_img - self.optimum) ** 2))
				stddev = np.std(best_img, axis=0)
				print("r:", r, ", stddev:", np.mean(stddev))

				if self.filepath:
					csv_writer.writerow([
						n_eval,
						max_n_eval,
						r,
						np.mean(stddev),
						g_loss,
						np.mean(gen_fitness),
						gen_fitness[ascending_indice][-1],
						best_fitness,
					])

				if self.filepath_distribution_after:
					os.makedirs(os.path.dirname("benchmark/distribution"), exist_ok=True)
					f_ = open(f"benchmark/distribution/{n_eval}.csv", mode = "w")
					csv_writer_ = csv.writer(f_)
					for _ in range(10):
						noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
						gen_imgs = self.generator.predict(noise)
						gen_imgs.reshape(-1, gen_imgs.shape[2])
						for row in gen_imgs.reshape(-1, gen_imgs.shape[2]):
							csv_writer_.writerow(row)
					f_.close()

		print(best_fitness)
		if self.filepath:
			f.close()

		if self.filepath_distribution_after:
			os.makedirs(os.path.dirname(self.filepath_distribution_after), exist_ok=True)
			f = open(self.filepath_distribution_after, mode = "w")
			csv_writer = csv.writer(f)
			for _ in range(10):
				noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
				gen_imgs = self.generator.predict(noise)
				gen_imgs.reshape(-1, gen_imgs.shape[2])
				for row in gen_imgs.reshape(-1, gen_imgs.shape[2]):
					csv_writer.writerow(row)
			f.close()

		return best_fitness

if __name__ == '__main__':
	def sphere(x):
		return -np.sum(x ** 2)

	def sphere_offset(x):
		return -np.sum((x - 0.5) ** 2)

	def ackley(x):
		x *= 32.768
		return -(20 - 20 * np.exp(- 0.2 * np.sqrt(1.0 / len(x) * np.sum(x ** 2))) +\
			np.e - np.exp(1.0 / len(x) * np.sum(np.cos(2 * np.pi * x))))

	def rastrigin(x):
		x *= 5.12
		return -10 * len(x) - np.sum(x ** 2) + 10 * np.sum(np.cos(2 * np.pi * x))

	evaluator = sphere_offset
	n_dim = 5

	nn = GreedyNN_MP_distribution(
		img_shape = (3, n_dim),
		evaluator = evaluator,
		optimum = [0.5] * n_dim,
		noise_dim = 1,
		fixed_noise=True,
		filepath_distribution_before=f"benchmark/distribution/{evaluator.__name__}_before.csv",
		filepath_distribution_after=f"benchmark/distribution/{evaluator.__name__}_after.csv")
	f = nn.train(max_n_eval=100, n_batch=10, batch_size=10)
	print(f)
