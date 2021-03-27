### -*-coding:utf-8-*-
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

# 提案法で近似を行わずにteacherを選ぶ
# greedynn_mp.pyとの差異はp[0] = p[1] = 0が無条件であることだけ
class GreedyNN_MP_Greedy():
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
		best_img = np.random.uniform(-1.0, 1.0, (self.img_shape[1]))
		best_fitness = np.NINF
		teacher_img = np.full((self.img_shape[0] - 1, self.img_shape[1]), np.NAN)
		teacher_fitness = np.full((self.img_shape[0] - 1), np.NINF)
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
			])

		while n_eval < max_n_eval:
			for iteration in range(n_batch):
				if not self.fixed_noise:
					noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
				gen_imgs = self.generator.predict(noise)
				gen_fitness = np.apply_along_axis(self.evaluator, 2, gen_imgs)

				ascending_indice = np.unravel_index(
					np.argsort(gen_fitness.flatten()), gen_fitness.shape)
				if gen_fitness[ascending_indice][-1] > best_fitness:
					best_fitness = gen_fitness[ascending_indice][-1]
					best_img = gen_imgs[ascending_indice][-1]

				fitness_pred_error = np.copy(gen_fitness)
				teacher_fitness_pred_error = np.copy(teacher_fitness)
				for i in range(gen_imgs.shape[2]):
					p = np.polyfit(gen_imgs[:, :, i].flatten(), gen_fitness.flatten(), 2)
					# 凸関数でないときにも近似を使わない
					# greedynn_mp.pyとの差異はここだけ
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
				y = np.tile(np.append([best_img], teacher_img, axis=0), (batch_size, 1, 1))
				g_loss = self.generator.train_on_batch(noise, y)

				n_eval += batch_size * self.img_shape[0]

				print ("eval:%d/%d, iter:%d/%d, [G loss: %f] [mean: %f best: %f]" %
					(n_eval, max_n_eval, iteration+1, n_batch,
					g_loss, np.mean(gen_fitness), best_fitness))
				print(f"b {np.append(np.array([best_fitness]), teacher_fitness)}")

				r = np.sqrt(np.sum((best_img - self.optimum) ** 2))
				stddev = np.std(gen_imgs, axis=0)
				print("r:", np.mean(r), ", stddev:", np.mean(stddev))

				if self.filepath:
					csv_writer.writerow([
						n_eval,
						max_n_eval,
						np.mean(r),
						np.mean(stddev),
						g_loss,
						np.mean(gen_fitness),
						gen_fitness[ascending_indice][-1],
						best_fitness,
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

	nn = GreedyNN_MP_Greedy(
		img_shape = (3, n_dim),
		evaluator = sphere_offset,
		optimum = [0.5] * n_dim,
		noise_dim = 1,
		fixed_noise=True)
	f = nn.train(max_n_eval=100, n_batch=10, batch_size=10)
	print(f)
