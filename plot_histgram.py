import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

def plot(filename):
	with open(filename, "r") as f:
		reader = csv.reader(f)
		data = []
		for row in reader:
			data.append(row[0])

		# ヒストグラムを出力
		plt.hist(data)
		plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--file", action = "store")
	args = parser.parse_args()

	plot(args.file)
