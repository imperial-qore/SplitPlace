from .mab_constants import *
import matplotlib.pyplot as plt

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = True
plt.rcParams['figure.figsize'] = 2, 1.2

colors = ['r', 'g', 'b']

def reduce(l):
	n = 10
	res = []
	low, high = [], []
	for i in range(0, len(l)):
		res.append(statistics.mean(l[max(0, i-n):min(len(l), i+n)]))
		low.append(min(l[max(0, i-n):min(len(l), i+n)])); high.append(max(l[max(0, i-n):min(len(l), i+n)]))
	return res, low, high

def plot(dats, labels, title):
	plt.xlabel('Intervals')
	plt.ylabel(label)
	for i, dat in enumerate(dats):
		plt.plot(range(len(dat)), dat, label=labels[i], color=colors[i], linewidth=1, marker='.')
	plt.legend(loc=1)
	plt.savefig('decider/MAB/graphs/'+title+'.pdf')
	plt.clf()

def plot_graphs(data):
	applications = ['MNIST', 'FashionMNIST', 'CIFAR100']
	layer_intervals = {}
	for app in applications:
		dat = [i[0][app.lower()] for i in data]
		layer_intervals[app] = dat
	epsilon = [i[1] for i in data]
	low_rewards, low_counts, high_rewards, high_counts = {}, {}, {}, {}
	all_arrays = [i[2] for i in data]
	choices = ['layer', 'semantic']
	for d, decision in enumerate(choices):
		low_rewards[decision] = [i[0][d] for i in all_arrays]
		low_counts[decision] = [i[1][d] for i in all_arrays]
		high_rewards[decision] = [i[2][d] for i in all_arrays]
		high_counts[decision] = [i[3][d] for i in all_arrays]
	plot([epsilon], ['Epsilon'], 'Epsilon')
	plot([layer_intervals[i] for i in applications], applications, 'Average Response Time')
	plot([low_rewards[i] for i in choices], choices, 'Rewards (low setting)')
	plot([low_counts[i] for i in choices], choices, 'Counts (low setting)')
	plot([high_rewards[i] for i in choices], choices, 'Rewards (high setting)')
	plot([high_counts[i] for i in choices], choices, 'Counts (high setting)')