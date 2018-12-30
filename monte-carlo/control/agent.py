"""
__author__ : kumar shubham
__date__   : 25-12-2018
__desc__   : file for creating the interaction of agent
"""
import numpy as np
from collections import defaultdict
import random 

class AgentControl:
	def __init__(self,gamma,noActSpace,noObsSpace):
		## gamma : discount factor - float
		## noActSpace : no of action -Int
		## noObsSpace : no of obsSpace - Int
		
		self.gamma = gamma
		self.noAction = noActSpace
		self.noObsSpace = noObsSpace
		
		############
		self.QFunction = {i:[0]*self.noAction for i in range(self.noObsSpace)} ## Q function for showcase
		self.tmpValue = {i:[0]*self.noAction for i in range(self.noObsSpace)} ## Q function over which calculation happens
		self.obsCount = {} ## this keep a count of obs

	def  action(self,observation,eps):
		
		### implementing epsillon greedy 
		if eps<random.random():
			return np.argmax(self.QFunction[observation])
		else:
			return np.random.choice(self.noAction)
	def episodicUpdate(self,listObsRew):
		## listObsRew : list of tuples of observation and reward
		## NOTE : we are implementing first visit monte carlo
		
		visited = {}
		counter = 0
		for obs,action,reward in listObsRew:
			## first visit
			if (obs,action) not in visited:
				visited[(obs,action)] = 1
				self.tmpValue[obs][action] +=self.finalRewardCalc(listObsRew[counter:])
				if (obs,action) not in self.obsCount:
					self.obsCount[(obs,action)]=1
				else:
					self.obsCount[(obs,action)]+=1
			else:
				pass
			counter+=1


		pass

	def updateQFn(self):
		## updating the value function with the average value
		for obs,action in self.obsCount.keys():
			self.QFunction[obs][action] = self.tmpValue[obs][action]/float(self.obsCount[(obs,action)])
		return 


	def finalRewardCalc(self,listObsRew):
		## return final reward for an episodic environment
		counter = 0
		output = 0
		for elm,action,reward in listObsRew:
			output += reward*np.power(self.gamma,counter)
			counter+=1
		return output

