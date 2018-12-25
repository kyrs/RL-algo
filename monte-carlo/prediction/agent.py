"""
__author__ : kumar shubham
__date__   : 25-12-2018
__desc__   : file for creating the interaction of agent
"""
import numpy as np
from collections import defaultdict
class AgentPrediction:
	def __init__(self,gamma,noActSpace,noObsSpace,policy):
		## gamma : discount factor - float
		## noActSpace : no of action -Int
		## noObsSpace : no of obsSpace - Int
		## policy : polic being followed - Dict

		self.gamma = gamma
		self.noAction = noActSpace
		self.noObsSpace = noObsSpace
		self.policy = policy
		############
		self.valueFunction = {i:0 for i in range(self.noObsSpace)} ## value function for showcase
		self.tmpValue = {i:0 for i in range(self.noObsSpace)} ## value function over which calculation happens
		self.obsCount = defaultdict(lambda:0) ## this keep a count of obs

	def  action(self,observation):
		## picking action as defined by the policy 
		return self.policy[observation]

	def episodicUpdate(self,listObsRew):
		## listObsRew : list of tuples of observation and reward
		## NOTE : we are implementing first visit monte carlo
		
		visited = {}
		counter = 0
		for obs,reward in listObsRew:
			## first visit
			if obs not in visited:
				visited[obs] = 1
				self.tmpValue[obs] +=self.finalRewardCalc(listObsRew[counter:])
				if obs not in self.obsCount:
					self.obsCount[obs]=1
				else:
					self.obsCount[obs]+=1
			else:
				pass
			counter+=1


		pass

	def updateValueFn(self):
		## updating the value function with the average value
		print(self.obsCount.keys())
		for obs in self.obsCount.keys():
			self.valueFunction[obs] = self.tmpValue[obs]/float(self.obsCount[obs])
		return 


	def finalRewardCalc(self,listObsRew):
		## return final reward for an episodic environment
		counter = 0
		output = 0
		for elm in listObsRew:
			reward = elm[1]
			output += reward*np.power(self.gamma,counter)
			counter+=1
		return output

