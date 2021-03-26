### -*-coding:utf-8-*-
import sys
import csv
import numpy as np
# import tensorflow as tf

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

class GreedyNN_MP_FIXED():
	def __init__(
			self,
			img_shape,
			evaluator,
			optimum,
			lr = 0.01,
			noise_dim = 100,
			fixed_noise = False,
			filepath = None):
		self.img_shape = img_shape
		self.noise_dim = noise_dim
		self.evaluator = evaluator
		self.optimum = optimum
		self.fixed_noise = fixed_noise
		self.filepath = filepath

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

	def train(self, max_n_eval, train_data, n_batch = 10, batch_size = 10):
		print('Number of batches:', n_batch)
		best_fitness = np.NINF
		best_img = np.random.uniform(-1.0, 1.0, (self.img_shape[1]))
		teacher_fitness = np.full((self.img_shape[0] - 1), np.NINF)
		teacher_img = np.full((self.img_shape[0] - 1, self.img_shape[1]), np.NAN)
		noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
		n_eval = 0

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
				"n_p",
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

				g_loss = self.generator.train_on_batch(noise, train_data)

				n_eval += batch_size * self.img_shape[0]

				# progress
				print ("eval:%d/%d, iter:%d/%d, [G loss: %f] [mean: %f best: %f]" %
					(n_eval, max_n_eval, iteration+1, n_batch,
					g_loss, np.mean(gen_fitness), best_fitness))
				print(f"b {best_fitness:.3} t xxx")

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
						self.img_shape[0],
					])

		print(best_fitness)
		if self.filepath:
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

	n_dim = 5
	train_data = np.array([[[0.5] * n_dim] * 3] * 10)

	nn = GreedyNN_MP_FIXED(
		img_shape = (3, n_dim),
		evaluator = sphere_offset,
		optimum = [0.5] * n_dim,
		noise_dim = 1,
		fixed_noise=True)
	f = nn.train(train_data=train_data, max_n_eval=100, n_batch=10, batch_size=10)
	print(f)
