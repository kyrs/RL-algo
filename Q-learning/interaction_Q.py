"""
__author__ : kumar shubham
__date__   : 25-12-2018
__desc__   : file for creating the interaction of agent
"""
import numpy as np 
import gym 

import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from matplotlib import style 

style.use("fivethirtyeight")


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)



def mainInt(env,agent):

	noEpisode =5000 ## episode to run
	QFnPrint = 400 ## after how many episode Q function is printed
	maxStepPrEps = 300 ## max step per episode

	count =0
	cost = []
	x = []
	for epCnt in range(1,noEpisode):
		observation  = env.reset()
		episodeList = list()
		for i in range(maxStepPrEps):
			## taking an action
			action = agent.action(observation,1/epCnt)
			## taking the needed action
			newObservation,reward,done,info = env.step(action)  			


			### next action
			## updating the value
			if done:
				break
			
			agent.updateQFn(oldObs=observation,oldAct=action,oldReward=reward,newObs=newObservation)
			observation = newObservation
			totalCost =agent.totalQVal()
			print(totalCost)
			cost.append(totalCost)
			x.append(count)

			count+=1
		if (epCnt%QFnPrint==0):
			env.render()
		else:
			pass
	ax1.plot(x,cost)
	plt.show()
