# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:59:36 2019
@author: tomsa
https://github.com/TomRSavage/ParticleSwarm
"""

import csv
import copy
import numpy as np
import numpy.random as rnd
import pso_util as util

def particleswarm(evaluator, bounds, p=60, c1=2.8, c2=1.3, vmax=1.5, max_n_eval=10000, filepath=None):
	'''
	DESCRIPTION
	see https://en.wikipedia.org/wiki/Particle_swarm_optimization

	INPUTS
	evaluator   :function to be optimized
	bounds      :bounds of each dimension in form [[x1,x2],[x3,x4]...]
	p           :number of particles
	c1          :adjustable parameter
	c2          :adjustable parameter
	vmax        :maximum particle velocity
	max_n_eval  :

	OUTPUTS
	swarm_best  : coordinates of optimal solution, with regards to exit
				  conditions
	'''
	print('Currently placing particles and giving them random \
	velocities...')
	d,particle_pos, particle_best, swarm_best, particle_velocity, local_best, pos_val =\
		util.initiation(evaluator,bounds,p) #initializing various arrays
	old_swarm_best=[0]*d
	c3=c1+c2
	K=2/(abs(2-c3-np.sqrt((c3**2)-(4*c3)))) #creating velocity weighting factor
	it_count = 0
	n_eval = 0
	particle_fitness = [None] * p

	if filepath:
		f = open(filepath, mode = "w")
		csv_writer = csv.writer(f)
		csv_writer.writerow([
			"n_eval",
			"max_n_eval",
			"dist_mean",
			"dist_stddev",
			"fitness_mean",
			"fitness_best",
			"fitness_best_so_far",
		])

	while n_eval < max_n_eval: #exit condition

		it_count+=1
		n_eval += p

		if it_count>1000: #every 1000 iterations...
						#create 'conflict' within the swarm and
						#give all particles random velocities
			print('Particles are too friendly! Creating conflict...')
			for j in range(p): #iterating ovre the number of particles
				particle_velocity[j]=[(rnd.uniform(-abs(bounds[i][1]-bounds[i][0]),\
					abs(bounds[i][1]-bounds[i][0]))) for i in range(d)]
					#adding random velocity values for each dimension
			it_count=0 #reset iteration count

		for i in range(p): #iterates over each particle
			rp,rg=rnd.uniform(0,1,2) #creates two random numbers between 0-
			particle_velocity[i,:]+=(c1*rp*(particle_best[i,:]-particle_pos[i,:]))
			particle_velocity[i,:]+=(c2*rg*(local_best[i,:]-particle_pos[i,:]))
			particle_velocity[i,:]=particle_velocity[i,:]*K
			if particle_velocity[i].any() > vmax: #is any velocity is greater than vmax
					particle_velocity[i,:]=vmax #set velocity to vmax
			#all of the above is regarding updating the particle's velocity
			#with regards to various parameters (local_best, p_best etc..)
			particle_pos[i,:]+=particle_velocity[i,:] #updating position

			util.withinbounds(bounds,particle_pos[i]) #if particle is out of bounds

			particle_fitness[i]=evaluator(particle_pos[i])

			if particle_fitness[i] < pos_val[i]:
				particle_best[i,:]=particle_pos[i,:] #checking if new best
				pos_val[i]=particle_fitness[i]
				f_swarm_best=evaluator(swarm_best)
				if particle_fitness[i] < f_swarm_best:
					old_swarm_best=swarm_best[:]
					swarm_best=copy.deepcopy(particle_best[i,:])
					print(f"{n_eval}/{max_n_eval}, {f_swarm_best}")

		local_best=util.local_best_get(particle_pos,pos_val,p)

		if filepath:
			csv_writer.writerow([
				n_eval,
				max_n_eval,
				np.mean(particle_pos),
				np.mean(np.std(particle_pos, axis=0)),
				-np.mean(particle_fitness),
				-sorted(particle_fitness)[0],
				-f_swarm_best,
			])

	if filepath:
		f.close()
	print('Optimum at: ',swarm_best,'\n','Function at optimum: ',evaluator(swarm_best))
	return evaluator(swarm_best)

if False:
	f=util.Rosenbrock
	dimensions=10
	dimension_bounds=[-2,2]
	bounds=[0]*dimensions #creating 5 dimensional bounds
	for i in range(dimensions):
		bounds[i]=dimension_bounds

	#creates bounds [[x1,x2],[x3,x4],[x5,x6]....]

	p=60 #shouldn't really change
	vmax=(dimension_bounds[1]-dimension_bounds[0])*0.75
	c1=2.8 #shouldn't really change
	c2=1.3 #shouldn't really change
	max_n_eval=10000

	particleswarm(f,bounds,p,c1,c2,vmax,max_n_eval)

if __name__ == "__main__" :
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

	evaluator = sphere_offset
	ndim = 20

	fitness = particleswarm(
		evaluator = lambda x: -evaluator(np.array(x, dtype=np.float)),
		bounds = [[-1, 1] for _ in range(ndim)],
		max_n_eval = int(1e5),
		filepath="hoge.csv")
