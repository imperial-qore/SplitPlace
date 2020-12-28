import matplotlib.pyplot as plt
import os
import pandas as pd 
import numpy as np
import torch
import random
import statistics

from sys import argv

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = True
plt.rcParams['figure.figsize'] = 2, 1.2

def reduce(l):
	n = 10
	res = []
	low, high = [], []
	for i in range(0, len(l)):
		res.append(statistics.mean(l[max(0, i-n):min(len(l), i+n)]))
		low.append(min(l[max(0, i-n):min(len(l), i+n)])); high.append(max(l[max(0, i-n):min(len(l), i+n)]))
	return res, low, high

class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def plot_accuracies(accuracy_list, filename):
	trainLoss = [i[0] for i in accuracy_list]
	testLoss = [i[1] for i in accuracy_list]
	testAcc = [i[2] for i in accuracy_list]
	if not os.path.exists('graphs'): os.mkdir('graphs')
	if not os.path.exists('graphs/'+filename): os.mkdir('graphs/'+filename)
	plt.xlabel('Epochs')
	plt.ylabel('Average Training Loss')
	plt.plot(range(len(trainLoss)), trainLoss, label='Average Training Loss', linewidth=1, linestyle='-', marker='.')
	plt.legend(loc=1)
	plt.savefig('graphs/'+filename+'/training-graph.pdf')
	plt.clf()
	plt.xlabel('Epochs')
	plt.ylabel('Average Testing Loss')
	plt.errorbar(range(len(testLoss)), testLoss, label='Average Testing Loss', alpha = 0.7,\
	    linewidth = 1, linestyle='dotted', marker='+')
	plt.legend(loc=4)
	plt.savefig('graphs/'+filename+'/testing-graph.pdf')
	plt.clf()
	plt.xlabel('Epochs')
	plt.ylabel('Average Testing Accuracy')
	plt.errorbar(range(len(testAcc)), testAcc, label='Average Testing Accuracy', alpha = 0.7,\
	    linewidth = 1, linestyle='dotted', marker='.')
	plt.legend(loc=4)
	plt.savefig('graphs/'+filename+'/testing-acc-graph.pdf')
	plt.clf()
	plt.xlabel('Epochs')
	plt.ylabel('Testing Loss')
	a, b, c = reduce(testLoss)
	b2, _, _ = reduce(b); c2, _, _ = reduce(c)
	plt.fill_between(np.arange(len(testLoss)), b2, c2, color='lightgreen', alpha=.5)
	plt.plot(a, label='Testing Loss', alpha = 0.7, color='g',\
	    linewidth = 1, linestyle='-')
	plt.savefig('graphs/'+filename+'/reduced.pdf')
