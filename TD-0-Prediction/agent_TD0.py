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
		
	def  action(self,observation):
		## picking action as defined by the policy 
		return self.policy[observation]

	def updateValueFn(self, newObs, oldObs,reward):
		## file for updating the value function based on TD
		## TD-0  obs[state] = obs[state] + learningRate[target- obs[state]]
		rewardOut =  self.alpha*( reward +self.gamma*self.valueFunction[newObs]-self.valueFunction[oldObs])
		self.valueFunction[oldObs] = self.valueFunction[oldObs] + reward
	

	def totalValue(self):
		## return the total cost in policy
		sumValue = np.sum(list(self.valueFunction.values()))
		return round(sumValue,4)