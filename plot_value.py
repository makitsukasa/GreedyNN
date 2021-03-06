import sys
import argparse
import csv
import glob
import numpy as np

def plot(filenames, xfield, yfields, yerrors, labels, xlabel, ylabel, log_scaled = False, x_max = None, hide_legend = False):
	if isinstance(filenames, str):
		filenames = [filenames]

	raw_datas = {}
	for filename in filenames:
		with open(filename, "r") as f:
			reader = csv.DictReader(f)
			fieldnames = next(reader).keys()
		with open(filename, "r") as f:
			data = {}
			for fieldname in fieldnames:
				data[fieldname] = []
			reader = csv.DictReader(f)
			for row in reader:
				for fieldname in fieldnames:
					data[fieldname].append(row[fieldname])
			base_file_name = filename.split("\\")[-1].replace(".csv", "")
			method_name = "_".join(base_file_name.split("_")[:-1])
			index = int(base_file_name.split("_")[-1])
			if not method_name in raw_datas:
				raw_datas[method_name] = {}
			raw_datas[method_name][index] = {}
			for fieldname in fieldnames:
				raw_datas[method_name][index][fieldname] = list(map(float, data[fieldname]))

	datas = {}
	counter = {}
	for method_name, method_data in raw_datas.items():
		datas[method_name] = {}
		counter[method_name] = {}
		for index, indexed_data in method_data.items():
			for fieldname, field_data in indexed_data.items():
				if not fieldname in datas[method_name].keys():
					datas[method_name][fieldname] = {}
					counter[method_name][fieldname] = {}
				for i in range(len(field_data)):
					if i in datas[method_name][fieldname].keys():
						datas[method_name][fieldname][i] += field_data[i]
						counter[method_name][fieldname][i] += 1
					else:
						datas[method_name][fieldname][i] = field_data[i]
						counter[method_name][fieldname][i] = 1

	for method_name, method_data in datas.items():
		for fieldname, field_data in method_data.items():
			if all([i.is_integer() for i in datas[method_name][fieldname].values()]):
				datas[method_name][fieldname] = np.array(list(map(
					lambda i: int(datas[method_name][fieldname][i]) // counter[method_name][fieldname][i],
					datas[method_name][fieldname].keys())))
			else:
				datas[method_name][fieldname] = np.array(list(map(
					lambda i: datas[method_name][fieldname][i] / counter[method_name][fieldname][i],
					datas[method_name][fieldname].keys())))

	for method_name, data in datas.items():
		indices = np.argsort(data[xfield])
		for fieldname, field_data in method_data.items():
			if not fieldname in datas[method_name].keys(): continue
			datas[method_name][fieldname] = datas[method_name][fieldname][indices]

	for method_name, data in datas.items():
		for i in range(len(yfields)):
			print(method_name, yfields[i])
			print(data[yfields[i]][-1])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--folder", action = "store")
	parser.add_argument("-x", "--xfield", action = "store")
	parser.add_argument("-y", "--yfields", action = "store", nargs='*')
	parser.add_argument("-e", "--yerrors", action = "store", nargs='*')
	parser.add_argument("--labels", action = "store", nargs='*', default=None)
	parser.add_argument("-l", "--log_scaled", action = "store_true")
	parser.add_argument("-m", "--max", action = "store")
	parser.add_argument("--hide_legend", action = "store_true")
	parser.add_argument("--xlabel", default = "個体の評価回数") # "Number of times individuals are evaluated"
	parser.add_argument("--ylabel", default = "") # "Objective function value"
	args = parser.parse_args()

	plot(
		glob.glob(f"{args.folder}/*.csv"),
		args.xfield,
		args.yfields,
		args.yerrors,
		args.labels,
		args.xlabel,
		args.ylabel,
		args.log_scaled,
		args.max,
		args.hide_legend)
