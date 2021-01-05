### -*-coding:utf-8-*-
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import sys
import numpy as np

# y is the answer. CheatyNN uses y.
# for compare with swapgan
class CheatyNN():
	def __init__(self, img_shape, n_gen_img, evaluator, y, z_dim = 100):
		self.img_shape = img_shape
		self.n_gen_img = n_gen_img
		self.y = y
		self.z_dim = z_dim
		self.evaluator = evaluator

		optimizer = Adam(0.0002, 0.5)

		# Generator model
		self.generator = self.build_generator()
		self.generator.compile(loss='mean_absolute_error', optimizer=optimizer)

	def build_generator(self):
		noise_shape = (self.z_dim,)
		model = Sequential()

		model.add(Dense(16, input_shape=noise_shape))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(16))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(16))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(np.prod(self.img_shape), activation='linear'))
		model.add(Reshape(self.img_shape))

		model.summary()

		return model

	def train(self, n_epoch, batch_size=64):
		n_batches = self.n_gen_img // batch_size
		print('Number of batches:', n_batches)
		best_fitness = np.NINF
		best_img = None
		for epoch in range(n_epoch):
			for iteration in range(n_batches):
				# ---------------------
				#  Generator learning
				# ---------------------
				# pickup images from generator
				noise = np.random.normal(0, 1, (batch_size, self.z_dim))
				gen_img = self.generator.predict(noise)
				gen_img_fitness = np.array([self.evaluator(d) for d in gen_img])

				# Train the generator
				y = self.y.repeat(len(gen_img)).reshape(batch_size, self.img_shape[0], 1)
				g_loss = self.generator.train_on_batch(noise, y)

				# swap
				best_index = np.argmax(gen_img_fitness)
				if gen_img_fitness[best_index] > best_fitness:
					best_fitness = gen_img_fitness[best_index]
					best_img = gen_img[best_index]

				# progress
				# print ("epoch:%d, iter:%d,  [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, iteration, d_loss[0], 100*d_loss[1], g_loss))
				print ("epoch:%d, iter:%d, [G loss: %f] [mean: %f best: %f]" %
					(epoch, iteration, g_loss, np.mean(gen_img_fitness), best_fitness))

				# print([self.evaluator(d) for d in gen_img], train_img_fitness[0])

		print(best_fitness)
		return best_fitness

if __name__ == '__main__':
	def sphere(x):
		return -np.sum(x ** 2)

	def sphere_offset(x):
		return -np.sum((x - 0.5) ** 2)

	nn = NN(
		img_shape = (10, 1),
		n_gen_img = 500,
		evaluator = sphere_offset,
		y = np.array([0.5 for _ in range(10)]),
		z_dim = 1)
	nn.train(n_epoch=200, batch_size=50)
