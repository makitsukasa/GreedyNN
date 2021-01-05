### -*-coding:utf-8-*-
import sys
import numpy as np

# for compare with swapgan
class RandGen():
	def __init__(self, img_shape, n_gen_img, evaluator, z_dim = 100):
		self.img_shape = img_shape
		self.n_gen_img = n_gen_img
		self.z_dim = z_dim
		self.evaluator = evaluator

	def train(self, n_epoch, batch_size=64):
		n_batches = self.n_gen_img // batch_size
		print('Number of batches:', n_batches)
		best_fitness = np.NINF
		best_img = None
		for epoch in range(n_epoch):
			for iteration in range(n_batches):
				# pickup images from generator
				gen_img = np.random.uniform(-1, 1, (self.n_gen_img, self.img_shape[0], self.img_shape[1]))
				gen_img_fitness = np.array([self.evaluator(d) for d in gen_img])

				# swap
				best_index = np.argmax(gen_img_fitness)
				if gen_img_fitness[best_index] > best_fitness:
					best_fitness = gen_img_fitness[best_index]
					best_img = gen_img[best_index]

				# progress
				# print ("epoch:%d, iter:%d,  [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, iteration, d_loss[0], 100*d_loss[1], g_loss))
				print ("epoch:%d, iter:%d, [mean: %f best: %f]" %
					(epoch, iteration, np.mean(gen_img_fitness), best_fitness))
				# print([self.evaluator(d) for d in gen_img], train_img_fitness[0])

		print(best_fitness)
		return best_fitness

if __name__ == '__main__':
	def sphere(x):
		return -np.sum(x ** 2)

	def sphere_offset(x):
		return -np.sum((x - 0.5) ** 2)

	randgen = RandGen(
		img_shape = (10, 1),
		n_gen_img = 500,
		evaluator = sphere_offset,
		z_dim = 1)
	randgen.train(n_epoch=100, batch_size=50)
