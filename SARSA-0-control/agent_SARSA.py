"""
__author__ : kumar shubham
__date__   : 30-12-2018
__desc__   : file for prediction in SARSA-0 environment
"""
import numpy as np
from collections import defaultdict
import random
class AgentPrediction_SARSA:
	def __init__(self,gamma,noActSpace,noObsSpace,alpha):
		## gamma : discount factor - float
		## noActSpace : no of action -Int
		## noObsSpace : no of obsSpace - Int
		## policy : polic being followed - Dict
		## alpha : learning step - Int 

		self.gamma = gamma
		self.noAction = noActSpace
		self.noObsSpace = noObsSpace
		self.alpha = alpha
		############
		self.QFunction = {i:[0]*self.noAction for i in range(self.noObsSpace)} ## Q function 
		
	def  action(self,observation,eps):
		### implementing epsillon greedy 
		if eps<random.random():
			return np.argmax(self.QFunction[observation])
		else:
			return np.random.choice(self.noAction)

	def updateQFn(self,oldObs,oldAct,oldReward,newObs,newAct ):
		## implementing the SARSA update 
		self.QFunction[oldObs][oldAct] = self.QFunction[oldObs][oldAct] + self.alpha* (oldReward+self.gamma*self.QFunction[newObs][newAct] -self.QFunction[oldObs][oldAct]) 

	def totalQVal(self):
		## return the total cost in policy
		sumValue=0
		for i in self.QFunction:
			sumValue+= np.sum(self.QFunction[i])
		return round(sumValue,4)