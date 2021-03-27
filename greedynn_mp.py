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

# 提案法 多点
class GreedyNN_MP():
	def __init__(
			self,
			img_shape,
			evaluator,
			optimum,
			lr = 0.01,
			noise_dim = 100,
			fixed_noise = False,
			filepath = None):
		"""
		Parameters
			img_shape   [0]:Generatorの出力個体数(2以上)，[1]:個体の次元数
			evaluator   評価関数 最大化問題
			optimum     大域的最適解 出力のみに使用しアルゴリズム本体には関与しない
			lr          Adamの学習率
			noise_dim   Generatorの入力の大きさ
			fixed_noise Generatorの入力を全世代で同じにする(True)か毎世代生成し直す(False)か
			filepath    出力のパス
		"""
		self.img_shape = img_shape
		self.noise_dim = noise_dim
		self.evaluator = evaluator
		self.optimum = optimum
		self.fixed_noise = fixed_noise
		self.filepath = filepath

		# Generatorを生成
		optimizer = Adam(lr)
		self.generator = self.build_generator()
		self.generator.compile(loss='mean_absolute_error', optimizer=optimizer)

	# Generatorを生成
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
			# np.prod(n_unit), activation=lambda x: tf.math.tanh(x/2), kernel_initializer=RandomUniform(-1,1)))
		model.add(Reshape(self.img_shape))
		model.summary()
		return model

	def train(self, max_n_eval, n_batch = 10, batch_size = 10):
		"""
		学習

		Parameters
			max_n_eval 目的関数の最大呼出回数
			n_batch    バッチ数
			batch_size バッチサイズ

		Variables
			best_img        今までで見つかった最良個体
			best_fitness    best_imgの評価値
			teacher_img     Generatorの教師データとなる個体のうちbest_img以外のもの
			teacher_fitness teacher_imgの評価値
			noise           Generatorの入力
			n_eval          今までの目的関数の呼出回数
			gen_imgs        生成した個体 (n_batch, img_shape[0], img_shape[1])
			gen_fitness     gen_imgsの評価値 (n_batch, img_shape[0])
		"""

		best_img = np.random.uniform(-1.0, 1.0, (self.img_shape[1]))
		best_fitness = np.NINF
		teacher_img = np.full((self.img_shape[0] - 1, self.img_shape[1]), np.NAN)
		teacher_fitness = np.full((self.img_shape[0] - 1), np.NINF)
		noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
		n_eval = 0

		if self.filepath:
			# csvのヘッダを書き込む
			f = open(self.filepath, mode = "w")
			csv_writer = csv.writer(f)
			csv_writer.writerow([
				"n_eval",
				"max_n_eval",
				"dist_r",              # best_imgとoptimumとの距離
				"dist_stddev",         # dist_rの標準偏差
				"train_loss",          # トレーニングの損失値
				"fitness_mean",        # 今世代に生成した個体の評価値の平均
				"fitness_best",        # 今世代に生成した個体の評価値の最大値
				"fitness_best_so_far", # これまでに生成した全ての個体の評価値の最大値
				"n_p",                 # Generatorの出力個体数
			])

		while n_eval < max_n_eval:
			for iteration in range(n_batch):
				# 個体を生成
				if not self.fixed_noise:
					# Generatorの入力を生成
					noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
				gen_imgs = self.generator.predict(noise) # 個体を生成
				gen_fitness = np.apply_along_axis(self.evaluator, 2, gen_imgs) # 評価

				# bestを更新
				ascending_indice = np.unravel_index(
					np.argsort(gen_fitness.flatten()), gen_fitness.shape)
				if gen_fitness[ascending_indice][-1] > best_fitness:
					best_fitness = gen_fitness[ascending_indice][-1]
					best_img = gen_imgs[ascending_indice][-1]

				# teacherの更新 ここから
				# 評価値の真値と近似値の誤差が大きい個体は局所解の近くにあり，これをteacherとする
				fitness_pred_error = np.copy(gen_fitness) # 出力した個体の「誤差」
				teacher_fitness_pred_error = np.copy(teacher_fitness) # teacher_imgの「誤差」
				for i in range(gen_imgs.shape[2]):
					p = np.polyfit(gen_imgs[:, :, i].flatten(), gen_fitness.flatten(), 2)
					if p[0] > 0:
						# 凸関数のときは近似がうまく機能していないだろうから近似を使わない
						p[0] = p[1] = 0
					y_pred = (p[0] * gen_imgs[:, :, i] ** 2 +
						p[1] * gen_imgs[:, :, i] + p[2]) / gen_imgs.shape[2]
					fitness_pred_error -= np.reshape(y_pred, fitness_pred_error.shape)
					t_pred = (p[0] * teacher_img[:, i] ** 2 +
						p[1] * teacher_img[:, i] + p[2]) / gen_imgs.shape[2]
					teacher_fitness_pred_error -= np.reshape(
						t_pred, teacher_fitness_pred_error.shape)

				# 今世代で生成した個体とteacher_imgを同じ配列に入れる
				vstacked_imgs = np.vstack((gen_imgs.reshape(-1, gen_imgs.shape[2]), teacher_img))
				vstacked_fitnesses = np.hstack((gen_fitness.flatten(), teacher_fitness))

				# vstacked_fitnessesを「誤差」が昇順になるようにソートした結果の配列のインデックス
				error_ascending_indice = np.argsort(np.hstack(
					(fitness_pred_error.flatten(), teacher_fitness_pred_error)))
				# bestは除外
				error_ascending_indice = error_ascending_indice[np.where(
					np.isfinite(vstacked_imgs).all(axis = 1) &
					(vstacked_imgs != best_img).all(axis = 1))]
				# 「誤差」が大きい順にteacherとする
				teacher_img = vstacked_imgs[error_ascending_indice][-teacher_img.shape[0]:]
				teacher_fitness = vstacked_fitnesses[error_ascending_indice][-teacher_img.shape[0]:]
				# teacherの更新 ここまで

				# Generatorの学習
				y = np.tile(np.append([best_img], teacher_img, axis=0), (batch_size, 1, 1))
				g_loss = self.generator.train_on_batch(noise, y)

				n_eval += batch_size * self.img_shape[0]

				# 出力
				print ("eval:%d/%d, iter:%d/%d, [G loss: %f] [mean: %f best: %f]" %
					(n_eval, max_n_eval, iteration+1, n_batch,
					g_loss, np.mean(gen_fitness), best_fitness))
				print(f"b {best_fitness:.3} t {teacher_fitness}")

				r = np.sqrt(np.sum((best_img - self.optimum) ** 2))
				stddev = np.std(best_img, axis=0)
				print("r:", r, ", stddev:", np.mean(stddev))

				if self.filepath:
					# csvに書き込む
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

	nn = GreedyNN_MP(
		img_shape = (3, n_dim),
		evaluator = sphere_offset,
		optimum = [0.5] * n_dim,
		noise_dim = 1,
		fixed_noise=True)
	f = nn.train(max_n_eval=100, n_batch=10, batch_size=10)
	print(f)
