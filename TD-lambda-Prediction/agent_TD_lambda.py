"""
__author__ : kumar shubham
__date__   : 28-12-2018
__desc__   : file for prediction in TD-0 environment
"""
import numpy as np
from collections import defaultdict
class AgentPrediction_TD:
	def __init__(self,gamma,noActSpace,noObsSpace,policy,alpha):
		## gamma : discount factor - float
		## noActSpace : no of action -Int
		## noObsSpace : no of obsSpace - Int
		## policy : polic being followed - Dict
		## alpha : learning step 

		self.gamma = gamma
		self.noAction = noActSpace
		self.noObsSpace = noObsSpace
		self.policy = policy
		self.alpha = alpha
		############
		self.valueFunction = {i:0 for i in range(self.noObsSpace)} ## value function for showcase
		self.eligibTrace = {i:0 for i in range(self.noObsSpace)} ## eligibility trace
	def  action(self,observation):
		## picking action as defined by the policy 
		return self.policy[observation]

	def updateValueFn(self,reward,newObs,oldObs):
		## file for updating the value function based on TD
		rewardOut =   reward +self.gamma*self.valueFunction[newObs]-self.valueFunction[oldObs]
		for key in self.valueFunction:
			self.valueFunction[key]+=self.eligibTrace[key]*self.alpha*rewardOut

	
	def eligibTraceUpd(self,newObs):
		## updating the eligibility trace

		for key in self.eligibTrace:
			## updating all the values 
			self.eligibTrace[key]*=self.alpha*self.gamma

		self.eligibTrace[newObs]+=1


	def totalValue(self):
		## return the total cost in policy
		sumValue = np.sum(list(self.valueFunction.values()))
		return round(sumValue,4)