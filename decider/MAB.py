from .SplitDecision import *
from .MAB.mab_constants import *
from .MAB.utils import *
import random
import numpy as np

class MABDecider(SplitDecision):
	def __init__(self, train=False):
		super().__init__()
		self.applications = ['mnist', 'fashionmnist', 'cifar100']
		layer_intervals = [5, 8, 15]
		self.average_layer_intervals = dict(zip(self.applications, layer_intervals))
		self.workflowids_checked = []
		self.epsilon = 0.95
		self.low_rewards, self.low_counts = np.zeros(2), np.zeros(2)
		self.high_rewards, self.high_counts = np.zeros(2), np.zeros(2)
		self.train = train
		if train: self.model = []
		if not train: self.load_model()

	def load_model(self):
		with open(SAVE_PATH, 'rb') as f:
			self.average_layer_intervals, self.epsilon, all_arrays = pickle.load(f)[-1]
			self.low_rewards, self.low_counts, self.high_rewards, self.high_counts = all_arrays

	def save_model(self):
		all_arrays = self.low_rewards, self.low_counts, self.high_rewards, self.high_counts
		self.model.append(self.average_layer_intervals, self.epsilon, all_arrays)
		with open(SAVE_PATH, 'wb') as f:
			pickle.dump(self.model, f)
		plot_graphs(self.model)

	def updateAverages(self):
		for WorkflowID in self.env.destroyedworkflows:
			if WorkflowID not in self.workflowids_checked:
				dict_ = self.destroyedworkflows[WorkflowID]
				workflow = dict_['application'].split('/')[1].split('_')[0]
				decision = dict_['application'].split('/')[1].split('_')[1]
				decision = 0 if decision == self.choices[0] else 1
				if decision == 0:
					intervals = dict_['destroyAt'] - sdict_['createAt']
					self.average_layer_intervals[workflow] = 0.1 * intervals + 0.9 * self.average_layer_intervals[workflow]

	def updateRewards(self):
		for WorkflowID in self.env.destroyedworkflows:
			if WorkflowID not in self.workflowids_checked:
				self.workflowids_checked.append(WorkflowID)
				dict_ = self.destroyedworkflows[WorkflowID]
				workflow = dict_['application'].split('/')[1].split('_')[0]
				decision = dict_['application'].split('/')[1].split('_')[1]
				decision = 0 if decision == self.choices[0] else 1
				intervals = dict_['destroyAt'] - dict_['createAt']
				sla = dict_['sla']
				sla_reward = 1 if intervals <= sla else 0
				acc_reward = dict_['result'][0]/dict_['result'][1]
				reward = Coeff_SLA * sla_reward + Coeff_Acc * acc_reward
				low = sla < self.average_layer_intervals[workflow]
				if low:
					self.low_counts[decision] += 1
					self.low_rewards[decision] = self.low_rewards[decision] + (reward - self.low_rewards[decision]) / self.low_counts[decision]
				else:
					self.high_counts[decision] += 1
					self.low_rewards[decision] = self.low_rewards[decision] + (reward - self.low_rewards[decision]) / self.low_counts[decision]

	def decision(self, workflowlist):
		self.updateAverages()
		self.updateRewards()
		decisions = []
		for _, _, sla, workflow in workflowlist:
			if self.train and random.random() < self.epsilon:
				decisions.append(random.choice(self.choices))
			else:
				low = sla < self.average_layer_intervals[workflow.lower()]
				if low:
					decision.append(self.choices[np.argmax(self.low_rewards)])
				else:
					decision.append(self.choices[np.argmax(self.high_rewards)])
		self.epsilon *= 0.99
		if self.train: self.save_model()
		return [self.choices[0]] * len(workflowlist)