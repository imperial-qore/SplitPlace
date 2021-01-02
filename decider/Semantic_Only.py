from .SplitDecision import *

class SemanticOnlyDecider(SplitDecision):
	def __init__(self):
		super().__init__()

	def decision(self, workflowlist):
		return [self.choices[1]] * len(workflowlist)