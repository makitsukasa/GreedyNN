# https://gist.github.com/tstreamDOTh/4af1d6b5a641deda16641181aa1e9ee8

import random
import csv
import numpy as np

#--- MAIN
class Particle:
	def __init__(self,num_dimensions,x0):
		self.num_dimensions=num_dimensions
		self.position_i=[]          # particle position
		self.velocity_i=[]          # particle velocity
		self.pos_best_i=[]          # best position individual
		self.err_best_i=-1          # best error individual
		self.err_i=-1               # error individual

		for i in range(0,self.num_dimensions):
			self.velocity_i.append(random.uniform(-1, 1))
			self.position_i.append(x0[i])
			if x0[i] == 1:
				exit()

	# evaluate current fitness
	def evaluate(self,evaluator):
		self.err_i=evaluator(self.position_i)

		# check to see if the current position is an individual best
		if self.err_i < self.err_best_i or self.err_best_i==-1:
			self.pos_best_i=self.position_i
			self.err_best_i=self.err_i

	# update new particle velocity
	def update_velocity(self,pos_best_g):
		c1 = 2.8        # cognative constant
		c2 = 1.3        # social constant
		# w = 2/(abs(2-(c1+c2)-np.sqrt(((c1+c2)**2)-(4*(c1+c2)))))       # constant inertia weight (how much to weigh the previous velocity)
		w = 0.73

		for i in range(0,self.num_dimensions):
			r1=random.uniform(0, 1)
			r2=random.uniform(0, 1)

			vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
			vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
			self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

	# update the particle position based off new velocity updates
	def update_position(self):
		for i in range(0,self.num_dimensions):
			self.position_i[i]=self.position_i[i]+self.velocity_i[i]

			# # adjust maximum position if necessary
			# if self.position_i[i]>bounds[i][1]:
			# 	self.position_i[i]=bounds[i][1]

			# # adjust minimum position if neseccary
			# if self.position_i[i] < bounds[i][0]:
			# 	self.position_i[i]=bounds[i][0]

def pso(evaluator,
		optimum,
		n_dim = 10,
		n_particles = 60,
		max_n_eval = 1000,
		filepath = None):

	x0 = [random.uniform(-1, 1) for _ in range(n_dim)]
	bounds = [(-1,1)] * n_dim

	err_best_g=-1                   # best error for group
	pos_best_g=[]                   # best position for group

	if filepath:
		f = open(filepath, mode = "w")
		csv_writer = csv.writer(f)
		csv_writer.writerow([
			"n_eval",
			"max_n_eval",
			"dist_r",
			"dist_stddev",
			"fitness_mean",
			"fitness_best",
			"fitness_best_so_far",
		])

	# establish the swarm
	swarm=[]
	for i in range(0,n_particles):
		swarm.append(Particle(n_dim, x0))

	# begin optimization loop
	n_eval=0
	while n_eval < max_n_eval:
		#print n_eval,err_best_g
		# cycle through particles in swarm and evaluate fitness
		for j in range(0,n_particles):
			swarm[j].evaluate(evaluator)

			# determine if current particle is the best (globally)
			if swarm[j].err_i < err_best_g or err_best_g == -1:
				pos_best_g=list(swarm[j].position_i)
				err_best_g=float(swarm[j].err_i)

		# cycle through swarm and update velocities and position
		for j in range(0,n_particles):
			swarm[j].update_velocity(pos_best_g)
			swarm[j].update_position()
		n_eval += n_particles

		r = np.sqrt(np.sum((np.array(pos_best_g) - optimum) ** 2))

		print(f"{n_eval}/{max_n_eval}, {r} {err_best_g}")

		if filepath:
			csv_writer.writerow([
				n_eval,
				max_n_eval,
				r,
				np.mean(np.std([swarm[j].position_i for j in range(n_particles)], axis=0)),
				-np.mean([swarm[j].err_i for j in range(n_particles)]),
				-sorted([swarm[j].err_i for j in range(n_particles)])[0],
				-err_best_g,
			])

	if filepath:
		f.close()
	# print final results
	print('FINAL:')
	print(pos_best_g)
	print(err_best_g)
	return err_best_g

if __name__ == "__main__":
	def sphere(x):
		return -np.sum(x ** 2)

	def sphere_offset(x):
		return -np.sum((x - 0.5) ** 2)

	def ackley(x):
		x *= 32.768
		return -20 + 20 * np.exp(- 0.2 * np.sqrt(1.0 / len(x) * np.sum(x ** 2)))\
			- np.e + np.exp(1.0 / len(x) * np.sum(np.cos(2 * np.pi * x)))

	def rastrigin(x):
		x *= 5.12
		return -10 * len(x) - np.sum(x ** 2) + 10 * np.sum(np.cos(2 * np.pi * x))

	n_dim = 20
	evaluator = ackley
	optimum = [0.5] * n_dim
	max_n_eval = 100000
	n_particles = n_dim * 100

	pso(lambda x: -evaluator(np.array(x, dtype=np.float)),
		optimum,
		n_dim,
		n_particles,
		max_n_eval)
