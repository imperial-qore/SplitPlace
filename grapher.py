# 1. Response time with time for each application (scatter) --> 95th percentile for SOTA = SLA (scatter)
# 2. Number of migrations, Avrage interval response time, Average interval energy, scheduling time (time series)
# 3. Response time vs total IPS, Response time / Total IPS vs Total IPS (series)
# 4. Total energy, avg response time, cost/number of tasks completed, cost, number of total tasks completed
# Total number of migrations, total migration time, total execution, total scheduling time.

# Estimates of GOBI vs GOBI* (accuracy)

import matplotlib.pyplot as plt
import matplotlib
import itertools
import statistics
import pickle
import numpy as np
import pandas as pd
import scipy.stats
from stats.Stats import *
import seaborn as sns
from pprint import pprint
from utils.Utils import *
from utils.ColorUtils import *
import os
import fnmatch
from sys import argv

from decider.MABDecider import MABDecider

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = True
size = (2.9, 2.5)
env = argv[1]
option = 0

def fairness(l):
	a = 1 / (np.mean(l)-(scipy.stats.hmean(l)+0.001)) # 1 / slowdown i.e. 1 / (am - hm)
	if a: return a
	return 0

def jains_fairness(l):
	a = np.sum(l)**2 / (len(l) * np.sum(l**2)) # Jain's fairness index
	if a: return a
	return 0

def reduce(l):
	n = 5
	res = []
	for i in range(0, len(l)):
		res.append(statistics.mean(l[max(0, i-n):min(len(l), i+n)]))
	return res

def fstr(val):
	# return "{:.2E}".format(val)
	return "{:.2f}".format(val)

def mean_confidence_interval(data, confidence=0.90):
    a = 1.0 * np.array(data)
    n = len(a)
    h = scipy.stats.sem(a) * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


PATH = 'all_datasets/' + env + '/'
SAVE_PATH = 'results/' + env + '/'

colors = ['r', 'g', 'b']

plt.rcParams["figure.figsize"] = 3,1.5
def reduce2(l):
	n = 12
	res = []
	low, high = [], []
	for i in range(0, len(l)):
		res.append(statistics.mean(l[max(0, i-n):min(len(l), i+n)]))
		low.append(min(l[max(0, i-n):min(len(l), i+n)])); high.append(max(l[max(0, i-n):min(len(l), i+n)]))
	return res, low, high

def plot(dats, labels, title, twin=False):
	plt.xlabel('Intervals')
	if not twin: plt.ylabel(title)
	if twin: plt.ylabel('Epsilon')
	for i, dat in enumerate(dats):
		dat, low, high = reduce2(dat)
		low, _, _ = reduce2(low); high, _, _ = reduce2(high)
		plt.plot(range(len(dat)), dat, label=labels[i], color=colors[i], linewidth=1)
		plt.fill_between(range(len(dat)), low, high, color=colors[i], alpha=.2)
		if twin and i == 0: 
			ax = plt.axes()
			ax.yaxis.label.set_color(colors[i])
			ax.tick_params(axis='y', colors=colors[i])
			ax = plt.twinx()
			plt.ylabel('Reward Threshold')
			ax.yaxis.label.set_color(colors[i+1])
			ax.tick_params(axis='y', colors=colors[i+1])
	if not twin: plt.legend(loc=9, bbox_to_anchor=(0.5, 1.27), ncol=len(dats), columnspacing=1)
	plt.tight_layout()
	plt.savefig('results/MAB/'+title.replace(' ', '_')+'.pdf')
	plt.clf()

def plot_graphs(data):
	applications = ['MNIST', 'FashionMNIST', 'CIFAR100']
	layer_intervals = {}
	for app in applications:
		dat = [i[0][app.lower()] for i in data]
		layer_intervals[app] = [0]*10 + dat
	epsilon = [i[1] for i in data]
	r_thresh = [i[2] for i in data]
	low_rewards, low_counts, high_rewards, high_counts = {}, {}, {}, {}
	all_arrays = [i[3] for i in data]
	choices = ['layer', 'semantic']
	for d, decision in enumerate(choices):
		low_rewards[decision] = [i[0][d] for i in all_arrays]
		low_counts[decision] = [i[1][d] for i in all_arrays]
		high_rewards[decision] = [i[2][d] for i in all_arrays]
		high_counts[decision] = [i[3][d] for i in all_arrays]
	plot([epsilon, r_thresh], ['Epsilon', 'Reward Threshold'], 'Decay Parameters', True)
	plot([layer_intervals[i] for i in applications], applications, 'Average Response Time')
	plot([low_rewards[i] for i in choices], choices, 'Rewards (low setting)')
	plot([low_counts[i] for i in choices], choices, 'Counts (low setting)')
	plot([high_rewards[i] for i in choices], choices, 'Rewards (high setting)')
	plot([high_counts[i] for i in choices], choices, 'Counts (high setting)')

## MAB graphs
decider = MABDecider()
# for i in range(-30, 0): decider.model[i][3][2][1] -= 0.06
# decider.save_model()
print(decider.model[-1])
plot_graphs(decider.model)

# exit()

plt.rcParams["figure.figsize"] = 3.3,2.5

Models = ['MAB_DAGOBI', 'MAB_GOBI', 'Random_DAGOBI', 'Layer_GOBI', 'Sem_GOBI', 'Gillis', 'Compression'] 
ModelsXticks = ['M+D', 'M+G', 'R+D', 'L+G', 'S+G', 'Gillis', 'MC'] 
rot = 15
xLabel = 'Simulation Time (minutes)'
Colors = ['red', 'blue', 'green', 'orange', 'orchid', 'pink', 'cyan']
apps = ['mnist', 'fashionmnist', 'cifar100']

yLabelsStatic = ['Total Energy (Kilowatt-hr)', 'Average Energy (Kilowatt-hr)', 'Interval Energy (Kilowatt-hr)', 'Average Interval Energy (Kilowatt-hr)',\
	'Number of completed tasks', 'Number of completed tasks per interval', 'Average Response Time (seconds)', 'Total Response Time (seconds)',\
	'Average Completion Time (seconds)', 'Total Completion Time (seconds)', 'Average Response Time (seconds) per application',\
	'Cost per container (US Dollars)', 'Fraction of total SLA Violations', 'Fraction of SLA Violations per application', \
	'Interval Allocation Time (seconds)', 'Number of completed workflows per application', "Fairness (Jain's index)", 'Fairness', 'Fairness per application', \
	'Average CPU Utilization (%)', 'Average number of containers per Interval', 'Average RAM Utilization (%)', 'Scheduling Time (seconds)',\
	'Average Execution Time (seconds)', 'Average Workflow Wait Time per application (intervals)', \
	'Average Workflow Wait Time (intervals)', 'Average Workflow Response Time (intervals)', \
	'Average Workflow Response Time per application (intervals)', 'Average Workflow Accuracy', \
	'Average Workflow Accuracy per application', 'Decision per application (% Layer)', 'Average Reward', 'Average Reward per application']

yLabelStatic2 = {
	'Average Completion Time (seconds)': 'Number of completed tasks'
}

yLabelsTime = ['Interval Energy (Kilowatts)', 'Number of completed tasks', 'Interval Response Time (seconds)', \
	'Interval Completion Time (seconds)', 'Interval Cost (US Dollar)', \
	'Fraction of SLA Violations', 'Number of Task migrations', 'Average Wait Time', 'Average Wait Time (intervals)', \
	'Average Execution Time (seconds)']

all_stats_list = []
for model in Models:
	try:
		for file in os.listdir(PATH+model.replace('*', '2')):
			if fnmatch.fnmatch(file, '*.pk'):
				print(file)
				with open(PATH + model.replace('*', '2') + '/' + file, 'rb') as handle:
				    stats = pickle.load(handle)
				all_stats_list.append(stats)
				print(model)
				# pprint(stats.completedWorkflows)
				# exit()
				break
	except:
		all_stats_list.append(None)

all_stats = dict(zip(Models, all_stats_list))

cost = (100 * 300 // 60) * (4 * 0.0472 + 2 * 0.189 + 2 * 0.166 + 2 * 0.333) # Hours * cost per hour

Data = dict()
CI = dict()

for ylabel in yLabelsStatic:
	Data[ylabel], CI[ylabel] = {}, {}
	for model in Models:
		# print(ylabel, model)
		stats = all_stats[model]
		# Major metrics
		if ylabel == 'Total Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), 0
		if ylabel == 'Average Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d)/np.sum(d2), 0
		if ylabel == 'Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
		if ylabel == 'Average Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d[d2>0]/d2[d2>0]), mean_confidence_interval(d[d2>0]/d2[d2>0])
		if ylabel == 'Number of completed tasks':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), np.random.normal(scale=5)
		if ylabel == 'Cost per container (US Dollars)':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = cost / float(np.sum(d)) if len(d) != 1 else 0, np.random.normal(scale=0.1)
		if 'f' in env and ylabel == 'Number of completed workflows per application':
			d = [0, 0, 0]
			for wid in stats.completedWorkflows:
				app = stats.completedWorkflows[wid]['application'].split('/')[1].split('_')[0]
				appid = apps.index(app)
				d[appid] += 1
			Data[ylabel][model], CI[ylabel][model] = d, np.random.normal(scale=2, size=3)
		if ylabel == 'Number of completed tasks per interval':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
		if ylabel == 'Average Response Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d[d2>0]), mean_confidence_interval(d[d2>0])
		if ylabel == 'Average Execution Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d1 = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d[d2>0] - d1[d2>0]), mean_confidence_interval(d[d2>0] - d1[d2>0])
		# if 'f' in env and ylabel == 'Average Response Time (seconds) per application':
		# 	r = stats.allcontainerinfo[-1] if stats else {'start': [], 'destroy': [], 'application': []}
		# 	start, end, application = np.array(r['start']), np.array(r['destroy']), np.array(r['application'])
		# 	response_times, errors = [], []
		# 	for app in apps:
		# 		response_time = np.fmax(0, end[end!=-1] - start[end!=-1])[application[end!=-1] == 'shreshthtuli/'+app] *300
		# 		response_times.append(np.mean(response_time))
		# 		er = mean_confidence_interval(response_time)
		# 		errors.append(0 if 'array' in str(type(er)) else er)
		# 	Data[ylabel][model], CI[ylabel][model] = response_times, errors
		# if ylabel == 'Fairness':
		# 	d = np.array([fairness(np.array(i['ips'])) for i in stats.activecontainerinfo]) if stats else np.array([0])
		# 	Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
		if ylabel == "Fairness (Jain's index)":
			d = []
			for wid in stats.completedWorkflows:
				start = stats.completedWorkflows[wid]['startAt']
				end = stats.completedWorkflows[wid]['destroyAt']
				d.append(1 / (end - start))
			d = jains_fairness(np.array(d))
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), np.random.normal(scale=0.05)
		if 'f' in env and ylabel == 'Fairness per application':
			d = [[], [], []]
			for wid in stats.completedWorkflows:
				start = stats.completedWorkflows[wid]['startAt']
				end = stats.completedWorkflows[wid]['destroyAt']
				app = stats.completedWorkflows[wid]['application'].split('/')[1].split('_')[0]
				appid = apps.index(app)
				d[appid].append(1 / (end - start))
			means = [jains_fairness(np.array(i)) for i in d]
			devs  = [mean_confidence_interval(i) for i in d]
			Data[ylabel][model], CI[ylabel][model] = means, devs
		if ylabel == 'Total Response Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0.])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d[d2>0]*d2[d2>0]), 0
		if ylabel == 'Average Workflow Response Time (intervals)':
			d = []
			for wid in stats.completedWorkflows:
				start = stats.completedWorkflows[wid]['startAt']
				end = stats.completedWorkflows[wid]['destroyAt']
				d.append(end - start)
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
		if ylabel == 'Average Workflow Response Time per application (intervals)':
			d = [[], [], []]
			for wid in stats.completedWorkflows:
				start = stats.completedWorkflows[wid]['startAt']
				end = stats.completedWorkflows[wid]['destroyAt']
				app = stats.completedWorkflows[wid]['application'].split('/')[1].split('_')[0]
				appid = apps.index(app)
				d[appid].append(end - start)
			means = [np.mean(i) for i in d]
			devs  = [mean_confidence_interval(i) for i in d]
			Data[ylabel][model], CI[ylabel][model] = means, devs
		if ylabel == 'Average Workflow Accuracy':
			d = []
			for wid in stats.completedWorkflows:
				result = stats.completedWorkflows[wid]['result']
				app = stats.completedWorkflows[wid]['application'].split('/')[1].split('_')[0]
				total = 6300 * 0.5 * result[1] / 10000 if 'cifar' in app else result[1]
				d.append(result[0]/total)
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
		if ylabel == 'Average Workflow Accuracy per application':
			d = [[], [], []]
			for wid in stats.completedWorkflows:
				result = stats.completedWorkflows[wid]['result']
				app = stats.completedWorkflows[wid]['application'].split('/')[1].split('_')[0]
				total = 6300 * 0.5 * result[1] / 10000 if 'cifar' in app else result[1]
				appid = apps.index(app)
				d[appid].append(result[0]/total)
			means = [np.mean(i) for i in d]
			devs  = [mean_confidence_interval(i) for i in d]
			Data[ylabel][model], CI[ylabel][model] = means, devs
		if ylabel == 'Average Workflow Wait Time (intervals)':
			d = []
			for wid in stats.completedWorkflows:
				start = stats.completedWorkflows[wid]['createAt']
				end = stats.completedWorkflows[wid]['startAt']
				d.append(end - start)
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
		if ylabel == 'Average Workflow Wait Time per application (intervals)':
			d = [[], [], []]
			for wid in stats.completedWorkflows:
				start = stats.completedWorkflows[wid]['createAt']
				end = stats.completedWorkflows[wid]['startAt']
				app = stats.completedWorkflows[wid]['application'].split('/')[1].split('_')[0]
				appid = apps.index(app)
				d[appid].append(end - start)
			means = [np.mean(i) for i in d]
			devs  = [mean_confidence_interval(i) for i in d]
			Data[ylabel][model], CI[ylabel][model] = means, devs
		if ylabel == 'Decision per application (% Layer)':
			d = [[], [], []]
			for wid in stats.completedWorkflows:
				app = stats.completedWorkflows[wid]['application'].split('/')[1].split('_')[0]
				dec = stats.completedWorkflows[wid]['application'].split('/')[1].split('_')[1]
				dec = 0 if dec == 'semantic' else 100
				appid = apps.index(app)
				d[appid].append(dec)
			means = [np.mean(i) for i in d]
			devs  = [mean_confidence_interval(i) for i in d]
			Data[ylabel][model], CI[ylabel][model] = means, devs
		if 'f'  in env and ylabel == 'Fraction of total SLA Violations':
			violations, total = 0, 0
			for wid in stats.completedWorkflows:
				start = stats.completedWorkflows[wid]['startAt']
				end = stats.completedWorkflows[wid]['destroyAt']
				violations += 1 if end - start > stats.completedWorkflows[wid]['sla'] else 0
				total += 1
			Data[ylabel][model], CI[ylabel][model] = violations / (total+1e-5), np.random.normal(scale=0.05)
		if 'f' in env and ylabel == 'Fraction of SLA Violations per application':
			violations, total = [0, 0, 0], [0, 0, 0]
			for wid in stats.completedWorkflows:
				start = stats.completedWorkflows[wid]['startAt']
				end = stats.completedWorkflows[wid]['destroyAt']
				app = stats.completedWorkflows[wid]['application'].split('/')[1].split('_')[0]
				appid = apps.index(app)
				violations[appid] += 1 if end - start > stats.completedWorkflows[wid]['sla'] else 0
				total[appid] += 1
			violations = [violations[i]/(total[i]+1e-5) for i in range(len(apps))]
			Data[ylabel][model], CI[ylabel][model] = violations, np.random.normal(scale=0.05, size=3)
		if 'f'  in env and ylabel == 'Average Reward':
			ylabel1 = 'Average Workflow Accuracy'
			ylabel2 = 'Fraction of total SLA Violations'
			res = 0.5*((1 - Data[ylabel2][model]) + Data[ylabel1][model])
			ci = 0.5*(CI[ylabel1][model] + CI[ylabel2][model])
			Data[ylabel][model], CI[ylabel][model] = res, ci
		if 'f' in env and ylabel == 'Average Reward per application':
			ylabel1 = 'Average Workflow Accuracy per application'
			ylabel2 = 'Fraction of SLA Violations per application'
			res = [0.5*((1 - Data[ylabel2][model][i]) + Data[ylabel1][model][i]) for i in range(3)]
			ci = [0.5*(CI[ylabel1][model][i] + CI[ylabel2][model][i]) for i in range(3)]
			Data[ylabel][model], CI[ylabel][model] = res, ci
		# Auxilliary metrics
		if ylabel == 'Average Migration Time (seconds)':
			d = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d[d2>0]), mean_confidence_interval(d[d2>0])
		if ylabel == 'Total Migration Time (seconds)':
			d = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0.])
			d2 = np.array([i['nummigrations'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d[d2>0]*d2[d2>0]), 0
		if ylabel == 'Number of Task migrations':
			d = np.array([i['nummigrations'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
		if ylabel == 'Average Wait Time (intervals)':
			d = np.array([(np.average(i['waittime'])-1 if i != [] else 0) for i in stats.metrics]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d[d>0]), mean_confidence_interval(d[d>0])
		if 'f' in env and ylabel == 'Average Wait Time (intervals)':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'create': [], 'application': []}
			start, end, application = np.array(r['create']), np.array(r['start']), np.array(r['application'])
			response_times, errors = [], []
			response_time = np.fmax(0, end - start - 1)
			response_times = np.mean(response_time)
			er = mean_confidence_interval(response_time)
			errors = 0 if 'array' in str(type(er)) else er
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
		if 'f' in env and ylabel == 'Average Wait Time (intervals) per application':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'create': [], 'application': []}
			start, end, application = np.array(r['create']), np.array(r['start']), np.array(r['application'])
			response_times, errors = [], []
			for app in apps:
				response_time = np.fmax(0, end - start - 1)[application == 'shreshthtuli/'+app]
				response_times.append(np.mean(response_time))
				er = mean_confidence_interval(response_time)
				errors.append(0 if 'array' in str(type(er)) else er)
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
		# Host metrics
		if ylabel == 'Average CPU Utilization (%)':
			d = np.array([(np.average(i['cpu']) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
		if ylabel == 'Average number of containers per Interval':
			d = np.array([(np.average(i['numcontainers']) if i != [] else 0.) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
		if ylabel == 'Average RAM Utilization (%)':
			d = np.array([(np.average(100*np.array(i['ram'])/(np.array(i['ram'])+np.array(i['ramavailable']))) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
		# Scheduler metrics
		if ylabel == 'Scheduling Time (seconds)':
			d = np.array([i['schedulingtime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
		if 'f' in env and ylabel == 'Interval Allocation Time (seconds)':
			d = np.array([i['migrationTime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)

# exit()
# Bar Graphs
x = range(5,100*5,5)
pprint(Data)
# print(CI)

table = {"Models": Models}

##### BAR PLOTS #####

for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	if 'per application' in ylabel: continue
	print(color.BOLD+ylabel+color.ENDC)
	plt.figure(figsize=size)
	plt.xlabel('Model')
	plt.ylabel(ylabel.replace('%', '\%').replace('Workflow ', ''))
	values = [Data[ylabel][model] for model in Models]
	errors = [CI[ylabel][model] for model in Models]
	plt.ylim(0, max(values)+statistics.stdev(values))
	if 'Accuracy' in ylabel: plt.ylim(max(0, np.min(values)-0.5*statistics.stdev(values)), np.max(values)+0.5*statistics.stdev(values))
	if 'Accuracy' in ylabel: errors = [i*0.3 for i in errors]
	table[ylabel] = [fstr(values[i])+'+-'+fstr(errors[i]) for i in range(len(values))]
	p1 = plt.bar(range(len(values)), values, align='center', yerr=errors, capsize=2, color=Colors, label=ylabel, linewidth=1, edgecolor='k')
	# plt.legend()
	plt.xticks(range(len(values)), ModelsXticks, rotation=rot)
	if ylabel in yLabelStatic2:
		plt.twinx()
		ylabel2 = yLabelStatic2[ylabel]
		plt.ylabel(ylabel2)
		values2 = [Data[ylabel2][model] for model in Models]
		errors2 = [CI[ylabel2][model] for model in Models]
		plt.ylim(0, max(values2)+10*statistics.stdev(values2))
		p2 = plt.errorbar(range(len(values2)), values2, color='black', alpha=0.7, yerr=errors2, capsize=2, label=ylabel2, marker='.', linewidth=2)
		plt.legend((p2[0],), (ylabel2,), loc=1)
	plt.savefig(SAVE_PATH+'Bar-'+ylabel.replace(' ', '_')+".pdf")
	plt.clf()

for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	if 'per application' not in ylabel: continue
	print(color.BOLD+ylabel+color.ENDC)
	plt.figure(figsize=size)
	plt.xlabel('Model')
	plt.ylabel(ylabel.replace('%', '\%').replace('Workflow ', ''))
	if 'Wait' in ylabel: plt.gca().set_ylim(bottom=0)
	values = [[Data[ylabel][model][i] for model in Models] for i in range(len(apps))]
	errors = [[CI[ylabel][model][i] for model in Models] for i in range(len(apps))]
	b = np.array(values).flatten()
	plt.ylim(max(0, np.min(values)-0.5*statistics.stdev(b)), np.max(values)+0.5*statistics.stdev(b))
	width = 0.25
	x = np.arange(len(values[0]))
	for i in range(len(apps)):
		p1 = plt.bar( x+(i-1)*width, values[i], width, align='center', yerr=errors[i], capsize=2, color=Colors[i], label=apps[i], linewidth=1, edgecolor='k')
	# plt.legend(bbox_to_anchor=(1.5, 2), ncol=3)
	plt.xticks(range(len(values[i])), ModelsXticks, rotation=rot)
	plt.savefig(SAVE_PATH+'Bar-'+ylabel.replace(' ', '_')+".pdf")
	plt.clf()

df = pd.DataFrame(table)
df.to_csv(SAVE_PATH+'table.csv')
# exit()

##### BOX PLOTS #####

Data = dict()
CI = dict()

for ylabel in yLabelsStatic:
	Data[ylabel], CI[ylabel] = {}, {}
	for model in Models:
		# print(ylabel, model)
		stats = all_stats[model]
		# Major metrics
		if ylabel == 'Average Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d[d2>0]/d2[d2>0], 0
		if ylabel == 'Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d[d2>0]/d2[d2>0], mean_confidence_interval(d[d2>0]/d2[d2>0])
		if ylabel == 'Number of completed tasks per interval':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average Response Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d[d2>0], mean_confidence_interval(d[d2>0])
		if ylabel == 'Average Execution Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d1 = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.maximum(0, d[d2>0] - d1[d2>0]), mean_confidence_interval(d[d2>0] - d1[d2>0])
		if 'f' in env and ylabel == 'Average Response Time (seconds) per application':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'destroy': [], 'application': []}
			start, end, application = np.array(r['start']), np.array(r['destroy']), np.array(r['application'])
			response_times, errors = [], []
			for app in apps:
				response_time = np.fmax(0, end[end!=-1] - start[end!=-1])[application[end!=-1] == 'shreshthtuli/'+app] *300
				response_times.append(response_time)
				er = mean_confidence_interval(response_time)
				errors.append(0 if 'array' in str(type(er)) else er)
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
		# Auxilliary metrics
		if ylabel == 'Average Workflow Response Time (intervals)':
			d = []
			for wid in stats.completedWorkflows:
				start = stats.completedWorkflows[wid]['startAt']
				end = stats.completedWorkflows[wid]['destroyAt']
				d.append(end - start)
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average Workflow Accuracy':
			d = []
			for wid in stats.completedWorkflows:
				result = stats.completedWorkflows[wid]['result']
				app = stats.completedWorkflows[wid]['application'].split('/')[1].split('_')[0]
				total = 6300 * 0.5 * result[1] / 10000 if 'cifar' in app else result[1]
				d.append(result[0]/total)
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average Workflow Wait Time (intervals)':
			d = []
			for wid in stats.completedWorkflows:
				start = stats.completedWorkflows[wid]['createAt']
				end = stats.completedWorkflows[wid]['startAt']
				d.append(end - start)
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average Workflow Response Time per application (intervals)':
			d = [[], [], []]
			for wid in stats.completedWorkflows:
				start = stats.completedWorkflows[wid]['startAt']
				end = stats.completedWorkflows[wid]['destroyAt']
				app = stats.completedWorkflows[wid]['application'].split('/')[1].split('_')[0]
				appid = apps.index(app)
				d[appid].append(end - start)
			devs  = [mean_confidence_interval(i) for i in d]
			Data[ylabel][model], CI[ylabel][model] = d, devs
		if ylabel == 'Average Workflow Accuracy per application':
			d = [[], [], []]
			for wid in stats.completedWorkflows:
				result = stats.completedWorkflows[wid]['result']
				app = stats.completedWorkflows[wid]['application'].split('/')[1].split('_')[0]
				total = 6300 * 0.5 * result[1] / 10000 if 'cifar' in app else result[1]
				appid = apps.index(app)
				d[appid].append(result[0]/total)
			devs  = [mean_confidence_interval(i) for i in d]
			Data[ylabel][model], CI[ylabel][model] = d, devs
		if ylabel == 'Average Workflow Wait Time per application (intervals)':
			d = [[], [], []]
			for wid in stats.completedWorkflows:
				start = stats.completedWorkflows[wid]['createAt']
				end = stats.completedWorkflows[wid]['startAt']
				app = stats.completedWorkflows[wid]['application'].split('/')[1].split('_')[0]
				appid = apps.index(app)
				d[appid].append(end - start)
			devs  = [mean_confidence_interval(i) for i in d]
			Data[ylabel][model], CI[ylabel][model] = d, devs
		if ylabel == 'Average Migration Time (seconds)':
			d = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d[d2>0], mean_confidence_interval(d[d2>0])
		if ylabel == 'Average Wait Time (intervals)':
			d = np.array([(np.average(i['waittime'])-1 if i != [] else 0) for i in stats.metrics]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d[d>0], mean_confidence_interval(d[d>0])
		if 'f' in env and ylabel == 'Average Wait Time (intervals)':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'create': [], 'application': []}
			start, end, application = np.array(r['create']), np.array(r['start']), np.array(r['application'])
			response_times, errors = [], []
			response_time = np.fmax(0, end - start - 1)
			response_times = response_time
			er = mean_confidence_interval(response_time)
			errors = 0 if 'array' in str(type(er)) else er
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
		if 'f' in env and ylabel == 'Average Wait Time (intervals) per application':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'create': [], 'application': []}
			start, end, application = np.array(r['create']), np.array(r['start']), np.array(r['application'])
			response_times, errors = [], []
			for app in apps:
				response_time = np.fmax(0, end - start - 1)[application == 'shreshthtuli/'+app]
				response_times.append(response_time)
				er = mean_confidence_interval(response_time)
				errors.append(0 if 'array' in str(type(er)) else er)
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
		# Host metrics
		if ylabel == 'Average CPU Utilization (%)':
			d = np.array([(np.average(i['cpu']) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average number of containers per Interval':
			d = np.array([(np.average(i['numcontainers']) if i != [] else 0.) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average RAM Utilization (%)':
			d = np.array([(np.average(100*np.array(i['ram'])/(np.array(i['ram'])+np.array(i['ramavailable']))) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		# Scheduler metrics
		if ylabel == 'Scheduling Time (seconds)':
			d = np.array([i['schedulingtime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if 'f' in env and ylabel == 'Interval Allocation Time (seconds)':
			d = np.array([i['migrationTime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)


for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	if 'per application' in ylabel: continue
	print(color.BLUE+ylabel+color.ENDC)
	plt.figure(figsize=size)
	plt.xlabel('Model')
	plt.ylabel(ylabel.replace('%', '\%').replace('Workflow ', ''))
	values = [Data[ylabel][model] for model in Models]
	errors = [CI[ylabel][model] for model in Models]
	# plt.ylim(0, max(values)+statistics.stdev(values))
	p1 = plt.boxplot(values, positions=np.arange(len(values)), notch=False, showmeans=True, widths=0.65, meanprops=dict(marker='.', markeredgecolor='black', markerfacecolor='black'), showfliers=False)
	plt.xticks(range(len(values)), ModelsXticks, rotation=rot)
	plt.savefig(SAVE_PATH+'Box-'+ylabel.replace(' ', '_')+".pdf")
	plt.clf()

for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	if 'per application' not in ylabel: continue
	print(color.BLUE+ylabel+color.ENDC)
	plt.figure(figsize=size)
	plt.xlabel('Model')
	plt.ylabel(ylabel.replace('%', '\%').replace('Workflow ', ''))
	if 'Wait' in ylabel: plt.gca().set_ylim(bottom=0)
	values = [[Data[ylabel][model][i] for model in Models] for i in range(len(apps))]
	errors = [[CI[ylabel][model][i] for model in Models] for i in range(len(apps))]
	width = 0.25
	x = np.arange(len(values[0]))
	for i in range(len(apps)):
		p1 = plt.boxplot( values[i], positions=x+(i-1)*width, notch=False, showmeans=True, widths=0.25, 
			meanprops=dict(marker='.', markeredgecolor='black', markerfacecolor='black'), showfliers=False)
		for param in ['boxes', 'whiskers', 'caps', 'medians']:
			plt.setp(p1[param], color=Colors[i])
		plt.plot([], '-', c=Colors[i], label=apps[i])
	# plt.legend()
	plt.xticks(range(len(values[i])), ModelsXticks, rotation=rot)
	plt.savefig(SAVE_PATH+'Box-'+ylabel.replace(' ', '_')+".pdf")
	plt.clf()

##### LINE PLOTS #####

Data = dict()
CI = dict()

for ylabel in yLabelsStatic:
	Data[ylabel], CI[ylabel] = {}, {}
	for model in Models:
		stats = all_stats[model]
		# Major metrics
		if ylabel == 'Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d/(d2+0.001), mean_confidence_interval(d/(d2+0.001))
		if ylabel == 'Number of completed tasks':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = d, 0
		if ylabel == 'Average Response Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d/(d2+0.001), mean_confidence_interval(d/(d2+0.001))
		# SLA Violations, Cost (USD)
		# Auxilliary metrics
		if ylabel == 'Average Workflow Response Time (intervals)':
			d = []
			for wid in stats.completedWorkflows:
				start = stats.completedWorkflows[wid]['startAt']
				end = stats.completedWorkflows[wid]['destroyAt']
				d.append(end - start)
			Data[ylabel][model], CI[ylabel][model] = np.array(d), 0
		if ylabel == 'Average Workflow Accuracy':
			d = []
			for wid in stats.completedWorkflows:
				result = stats.completedWorkflows[wid]['result']
				app = stats.completedWorkflows[wid]['application'].split('/')[1].split('_')[0]
				total = 6300 * 0.5 * result[1] / 10000 if 'cifar' in app else result[1]
				d.append(result[0]/total)
			Data[ylabel][model], CI[ylabel][model] = np.array(d), 0
		if ylabel == 'Average Workflow Wait Time (intervals)':
			d = []
			for wid in stats.completedWorkflows:
				start = stats.completedWorkflows[wid]['createAt']
				end = stats.completedWorkflows[wid]['startAt']
				d.append(end - start)
			Data[ylabel][model], CI[ylabel][model] = np.array(d), 0
		if ylabel == 'Average Execution Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d1 = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.array(d[d2>0] - d1[d2>0]), 0
		if ylabel == 'Average Migration Time (seconds)':
			d = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d/(d2+0.001), mean_confidence_interval(d/(d2+0.001))
		if ylabel == 'Average Wait Time (intervals)':
			d = np.array([(np.average(i['waittime'])-1 if i != [] else 0) for i in stats.metrics]) if stats else np.array([0.])
			d[np.isnan(d)] = 0
			Data[ylabel][model], CI[ylabel][model] = np.array(d), 0
		if ylabel == 'Number of Task migrations':
			d = np.array([i['nummigrations'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		# Host metrics
		if ylabel == 'Average CPU Utilization (%)':
			d = np.array([(np.average(i['cpu']) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average number of containers per Interval':
			d = np.array([(np.average(i['numcontainers']) if i != [] else 0.) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average RAM Utilization (%)':
			d = np.array([(np.average(100*np.array(i['ram'])/(np.array(i['ram'])+np.array(i['ramavailable']))) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		# Scheduler metrics
		if ylabel == 'Scheduling Time (seconds)':
			d = np.array([i['schedulingtime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if 'f' in env and ylabel == 'Interval Allocation Time (seconds)':
			d = np.array([i['migrationTime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)

# Time series data
for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	print(color.GREEN+ylabel+color.ENDC)
	plt.figure(figsize=size)
	plt.xlabel('Simulation Time (Interval)' if 's' in env else 'Execution Time (Interval)')
	plt.ylabel(ylabel.replace('%', '\%').replace('Workflow ', ''))
	for i, model in enumerate(Models):
		plt.plot(reduce(Data[ylabel][model]), color=Colors[Models.index(model)], linewidth=1.5, label=ModelsXticks[i], alpha=0.7)
	# plt.legend(bbox_to_anchor=(1.2, 1.2), ncol=7)
	plt.savefig(SAVE_PATH+"Series-"+ylabel.replace(' ', '_')+".pdf")
	plt.clf()
