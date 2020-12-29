from .SplitDecision import *

class SemanticOnlyDecider(Scheduler):
	def __init__(self):
		super().__init__()

	def decision(self, workflowlist):
		return [self.choices[1]] * len(workflowlist)