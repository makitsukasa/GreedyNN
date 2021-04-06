### -*-coding:utf-8-*-

# https://github.com/triwave33/GAN/blob/master/GAN/dcgan/dcgan.py
# please see https://qiita.com/triwave33/items/1890ccc71fab6cbca87e


import sys
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

# GANを用いた最適化法
# 良い個体を訓練データとする．生成した個体が良い個体だったら適宜訓練データとする個体を入れ替える．
class SwapGAN():
	def __init__(self, img_shape, pop_img, evaluator, noise_dim = 100, fixed_noise = False):
		"""
		Parameters
			img_shape Generatorの出力の大きさ
			pop_img   初期個体
			evaluator   評価関数 最大化問題
			noise_dim   Generatorの入力の大きさ
			fixed_noise Generatorの入力を全世代で同じにする(True)か毎世代生成し直す(False)か
		"""
		self.img_shape = img_shape
		self.pop_img = pop_img
		self.noise_dim = noise_dim
		self.evaluator = evaluator
		self.fixed_noise = fixed_noise

		optimizer = Adam(0.0002, 0.5)

		self.discriminator = self.build_discriminator()
		self.discriminator.compile(
			loss='binary_crossentropy',
			optimizer=optimizer,
			metrics=['accuracy'])

		self.generator = self.build_generator()

		self.combined = self.build_combined()
		self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

	# Generatorを生成 コンストラクタからのみ呼ばれる
	def build_generator(self):
		noise_shape = (self.noise_dim,)
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

	# Discriminatorを生成 コンストラクタからのみ呼ばれる
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

	# GeneratorとDiscriminatorを繋げたモデルを生成 コンストラクタからのみ呼ばれる
	def build_combined(self):
		self.discriminator.trainable = False
		model = Sequential([self.generator, self.discriminator])
		return model

	def train(self, max_n_eval, n_batch, batch_size=128):
		"""
		学習

		Parameters
			max_n_eval 目的関数の最大呼出回数
			n_batch    バッチ数
			batch_size バッチサイズ

		Variables
			n_epoch    エポック数
			self.pop_img      popの個体 ソートしてある
			pop_size          popの個体数
			pop_fitness       popの評価値
			halved_batch_size batch_size // 2
			noise             Generatorの入力
		"""

		n_epoch = max_n_eval * 2 // (n_batch * batch_size)
		pop_size = self.pop_img.shape[0]
		halved_batch_size = batch_size // 2
		pop_fitness = np.array([self.evaluator(d) for d in self.pop_img])
		noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
		n_eval = len(pop_fitness)

		# popをソート
		indices = np.argsort(-pop_fitness)
		self.pop_img = self.pop_img[indices]
		pop_fitness = pop_fitness[indices]

		for epoch in range(n_epoch):
			for iteration in range(n_batch):
				# Discriminatorの学習 ここから
				# Generatorでhalved_batch_size個の個体を生成
				if not self.fixed_noise:
					noise = np.random.normal(0, 1, (halved_batch_size, self.noise_dim))
				gen_img = self.generator.predict(noise[:halved_batch_size])

				# popからランダムにhalved_batch_size個の個体を選び出す
				idx = np.random.randint(0, self.pop_img.shape[0], halved_batch_size)
				imgs = self.pop_img[idx]

				# 学習 訓練データと生成されたデータで分けて学習を行っても影響はないらしい
				d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((halved_batch_size, 1)))
				d_loss_fake = self.discriminator.train_on_batch(gen_img, np.zeros((halved_batch_size, 1)))
				# Discriminatorのlossは上2つの平均とする
				d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
				# Discriminatorの学習 ここまで

				# Generatorの学習 ここから
				if not self.fixed_noise:
					noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
				# ラベル付け
				valid_y = np.array([1] * batch_size)
				# 学習
				g_loss = self.combined.train_on_batch(noise, valid_y)
				# Generatorの学習 ここまで

				# 良い個体が生成されたときは保持しておく個体と入れ替え
				self.pop_img = np.append(self.pop_img, gen_img, axis = 0)
				gen_img_fitness = [self.evaluator(d) for d in gen_img]
				pop_fitness = np.append(pop_fitness, gen_img_fitness)
				indices = np.argsort(-pop_fitness)[:pop_size]
				self.pop_img = self.pop_img[indices]
				pop_fitness = pop_fitness[indices]

				n_eval += len(gen_img_fitness)

				# 出力
				# print ("epoch:%d, iter:%d,  [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, iteration, d_loss[0], 100*d_loss[1], g_loss))
				print ("epoch:%d, iter:%d, %d [D loss: %f, acc.: %.2f%%] [G loss: %f] [mean: %f best: %f]" %
					(epoch, iteration, n_eval, d_loss[0], 100*d_loss[1], g_loss, np.mean(gen_img_fitness), pop_fitness[0]))

				# print([self.evaluator(d) for d in gen_img], pop_fitness[0])

		# print(pop_fitness[0])
		return pop_fitness[0]

if __name__ == '__main__':
	def sphere(x):
		return -np.sum(x ** 2)

	def sphere_offset(x):
		return -np.sum((x - 0.5) ** 2)

	gan = SwapGAN(
		img_shape = (10, 1),
		pop_img = np.random.uniform(-1.0, 1.0, (100, 10, 1)),
		evaluator = sphere_offset,
		noise_dim = 1)
	f = gan.train(n_epoch=20, batch_size=100)
	print(f)
