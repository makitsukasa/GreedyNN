### -*-coding:utf-8-*-
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import sys
import numpy as np

class SwapGAN():
	def __init__(self, img_shape, train_img, evaluator, z_dim = 100, fixed_noise = False):
		self.img_shape = img_shape
		self.train_img = train_img
		self.z_dim = z_dim
		self.evaluator = evaluator
		self.fixed_noise = fixed_noise

		optimizer = Adam(0.0002, 0.5)

		# discriminator model
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(
			loss='binary_crossentropy',
			optimizer=optimizer,
			metrics=['accuracy'])

		# Generator model
		self.generator = self.build_generator()
		# we don't need to compile generator because generator learns with descriminator as 'combined'
		#self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

		# we can define combined model by two ways (results are same)
		# please see https://qiita.com/triwave33/items/1890ccc71fab6cbca87e
		# sorry in Japanese
		self.combined = self.build_combined1()
		#self.combined = self.build_combined2()
		self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

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

	def build_discriminator(self):
		model = Sequential()
		if len(self.img_shape) != 1:
			model.add(Flatten(input_shape=self.img_shape))
		model.add(Dense(16))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(16))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(1, activation='sigmoid'))
		model.summary()

		return model

	def build_combined1(self):
		self.discriminator.trainable = False
		model = Sequential([self.generator, self.discriminator])
		return model

	def build_combined2(self):
		z = Input(shape=(self.z_dim,))
		img = self.generator(z)
		self.discriminator.trainable = False
		valid = self.discriminator(img)
		model = Model(z, valid)
		model.summary()
		return model

	def train(self, n_epoch, batch_size=128):
		# Number of batches is doubled because half of the data is from the Generator.
		n_train_img = self.train_img.shape[0]
		halved_batch_size = batch_size // 2
		n_batches = n_train_img // halved_batch_size
		print('Number of batches:', n_batches)
		train_img_fitness = np.array([self.evaluator(d) for d in self.train_img])
		indices = np.argsort(-train_img_fitness)
		self.train_img = self.train_img[indices]
		train_img_fitness = train_img_fitness[indices]
		noise = np.random.normal(0, 1, (batch_size, self.z_dim))

		for epoch in range(n_epoch):
			for iteration in range(n_batches):
				# ---------------------
				#  Discriminator learning
				# ---------------------
				# pickup images (half-batch size) from generator
				if not self.fixed_noise:
					noise = np.random.normal(0, 1, (halved_batch_size, self.z_dim))
				gen_img = self.generator.predict(noise[:halved_batch_size])
				print(gen_img.shape)
				exit()

				# pickup images (half-batch size) from dataset
				idx = np.random.randint(0, self.train_img.shape[0], halved_batch_size)
				imgs = self.train_img[idx]

				# learn discriminator
				# learning discriminator with real-data and fake-data seperately
				d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((halved_batch_size, 1)))
				d_loss_fake = self.discriminator.train_on_batch(gen_img, np.zeros((halved_batch_size, 1)))
				# average each loss
				d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

				# ---------------------
				#  Generator learning
				# ---------------------
				if not self.fixed_noise:
					noise = np.random.normal(0, 1, (batch_size, self.z_dim))

				# make label (gen data: 1)
				valid_y = np.array([1] * batch_size)

				# Train the generator
				g_loss = self.combined.train_on_batch(noise, valid_y)

				# swap
				self.train_img = np.append(self.train_img, gen_img, axis = 0)
				gen_img_fitness = [self.evaluator(d) for d in gen_img]
				train_img_fitness = np.append(train_img_fitness, gen_img_fitness)
				indices = np.argsort(-train_img_fitness)[:n_train_img]
				self.train_img = self.train_img[indices]
				train_img_fitness = train_img_fitness[indices]

				# progress
				# print ("epoch:%d, iter:%d,  [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, iteration, d_loss[0], 100*d_loss[1], g_loss))
				print ("epoch:%d, iter:%d,  [D loss: %f, acc.: %.2f%%] [G loss: %f] [mean: %f best: %f]" %
					(epoch, iteration, d_loss[0], 100*d_loss[1], g_loss, np.mean(gen_img_fitness), train_img_fitness[0]))

				# print([self.evaluator(d) for d in gen_img], train_img_fitness[0])

		# print(train_img_fitness[0])
		return train_img_fitness[0]

if __name__ == '__main__':
	def sphere(x):
		return -np.sum(x ** 2)

	def sphere_offset(x):
		return -np.sum((x - 0.5) ** 2)

	gan = SwapGAN(
		img_shape = (10, 1),
		train_img = np.random.uniform(-1.0, 1.0, (1000, 10, 1)),
		evaluator = sphere_offset,
		z_dim = 1)
	f = gan.train(n_epoch=20, batch_size=100)
	print(f)
