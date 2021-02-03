# -*- coding: utf-8 -*-

import numpy as np
import random

#評価関数: z = x^2 + y^2
def evaluator(x, y):
	z = x * x + y * y
	return z

#粒子の位置の更新を行う関数
def update_position(x, y, vx, vy):
	new_x = x + vx
	new_y = y + vy
	return new_x, new_y

#粒子の速度の更新を行う関数
def update_velocity(x, y, vx, vy, p, g, w=0.5, ro_max=0.14):
	#パラメーターroはランダムに与える
	ro1 = random.uniform(0, ro_max)
	ro2 = random.uniform(0, ro_max)
	#粒子速度の更新を行う
	new_vx = w * vx + ro1 * (p["x"] - x) + ro2 * (g["x"] - x)
	new_vy = w * vy + ro1 * (p["y"] - y) + ro2 * (g["y"] - y)
	return new_vx, new_vy

def main(max_n_eval = 1000):
	n_particle = 60  #粒子の数
	x_min, x_max = -1, 1
	y_min, y_max = -1, 1
	#粒子位置, 速度, パーソナルベスト, グローバルベストの初期化を行う
	ps = [{"x": random.uniform(x_min, x_max),
		"y": random.uniform(y_min, y_max)} for i in range(n_particle)]
	vs = [{"x": 0.0, "y": 0.0} for i in range(n_particle)]
	p_best_positions = list(ps)
	p_best_scores = [evaluator(p["x"], p["y"]) for p in ps]
	g_best_index = np.argmin(p_best_scores)
	g_best_position = p_best_positions[g_best_index]
	n_eval = n_particle

	while n_eval < max_n_eval:
		for n in range(n_particle):
			x, y = ps[n]["x"], ps[n]["y"]
			vx, vy = vs[n]["x"], vs[n]["y"]
			p = p_best_positions[n]
			#粒子の位置の更新を行う
			new_x, new_y = update_position(x, y, vx, vy)
			ps[n] = {"x": new_x, "y": new_y}
			#粒子の速度の更新を行う
			new_vx, new_vy = update_velocity(
				new_x, new_y, vx, vy, p, g_best_position)
			vs[n] = {"x": new_vx, "y": new_vy}
			#評価値を求め, パーソナルベストの更新を行う
			score = evaluator(new_x, new_y)
			if score < p_best_scores[n]:
				p_best_scores[n] = score
				p_best_positions[n] = {"x": new_x, "y": new_y}
		#グローバルベストの更新を行う
		g_best_index = np.argmin(p_best_scores)
		g_best_position = p_best_positions[g_best_index]

		print(p_best_scores[g_best_index])

		n_eval += n_particle
	#最適解
	print(g_best_position)
	print(min(p_best_scores))

if __name__ == '__main__':
	main()
