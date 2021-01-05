### -*-coding:utf-8-*-
from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

class GAN():
	def __init__(self, img_shape, train_data):
		self.img_shape = img_shape
		self.train_data = train_data
		self.z_dim = 100

		optimizer = Adam(0.0002, 0.5)

		# discriminator model
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(
			loss='binary_crossentropy',
			optimizer=optimizer,
			metrics=['accuracy'])

		# Generator model
		self.generator = self.build_generator()
		# we don't need to compile generator because generator learns with descriminatorgenerator as 'combined'
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

		model.add(Dense(256, input_shape=noise_shape))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(1024))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(np.prod(self.img_shape), activation='tanh'))
		model.add(Reshape(self.img_shape))

		model.summary()

		return model

	def build_discriminator(self):
		model = Sequential()
		if len(self.img_shape) != 1:
			model.add(Flatten(input_shape=self.img_shape))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(256))
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

	def train(self, epochs, batch_size=128, save_interval=50):

		# Number of batches is doubled because half of the data is from the Generator.
		halved_batch_size = batch_size // 2
		num_batches = self.train_data.shape[0] // halved_batch_size
		print('Number of batches:', num_batches)

		for epoch in range(epochs):
			for iteration in range(num_batches):
				# ---------------------
				#  Discriminator learning
				# ---------------------
				# pickup images (half-batch size) from generator
				noise = np.random.normal(0, 1, (halved_batch_size, self.z_dim))
				gen_imgs = self.generator.predict(noise)

				# pickup images (half-batch size) from dataset
				idx = np.random.randint(0, self.train_data.shape[0], halved_batch_size)
				imgs = self.train_data[idx]

				# learn discriminator
				# learning discriminator with real-data and fake-data seperately
				d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((halved_batch_size, 1)))
				d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((halved_batch_size, 1)))
				# average each loss
				d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

				# ---------------------
				#  Generator learning
				# ---------------------
				noise = np.random.normal(0, 1, (batch_size, self.z_dim))

				# make label (gen data: 1)
				valid_y = np.array([1] * batch_size)

				# Train the generator
				g_loss = self.combined.train_on_batch(noise, valid_y)

				# progress
				print ("epoch:%d, iter:%d,  [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, iteration, d_loss[0], 100*d_loss[1], g_loss))

			# save images
			if save_interval and (epoch % save_interval == 0):
				self.save_imgs(epoch)

	def save_imgs(self, epoch):
		# row,col
		r, c = 5, 5

		noise = np.random.normal(0, 1, (r * c, self.z_dim))
		gen_imgs = self.generator.predict(noise)

		# rescale [-1, 1] to [0, 1]
		gen_imgs = 0.5 * gen_imgs + 0.5

		fig, axs = plt.subplots(r, c)
		cnt = 0
		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
				axs[i,j].axis('off')
				cnt += 1
		fig.savefig("images/mnist_%d.png" % epoch)
		plt.close()

if __name__ == '__main__':
	from keras.datasets import mnist

	(train_data, _), (_, _) = mnist.load_data()
	train_data = (train_data.astype(np.float32) - 127.5) / 127.5 # normalize
	train_data = np.expand_dims(train_data, axis=3)

	gan = GAN((28, 28, 1), train_data)
	gan.train(epochs=1, batch_size=32, save_interval=1)
