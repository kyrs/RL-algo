"""
__author__ : kumar shubham
__date__   : 25-12-2018
__desc__   : file for creating the interaction of agent
"""
import numpy as np 
import gym 

def mainInt(env,agent):

	noEpisode =5000 ## episode to run
	valueFnUpd = 400 ## after how many episode value function is updated
	maxStepPrEps = 300 ## max step per episode

	for epCnt in range(noEpisode):
		observation  = env.reset()
		episodeList = list()
		for i in range(maxStepPrEps):
			## taking an action
			action = agent.action(observation)
			## taking the needed action
			observation,reward,done,info = env.step(action)  
			episodeList.append((observation,reward))
			if done:
				## episode reached
				break

		agent.episodicUpdate(episodeList)

		if (epCnt% valueFnUpd==0):
			agent.updateValueFn()
			env.render()
			print ("======Value fn after : %d epoch cnt======== "%(epCnt))
			print(agent.valueFunction)
			print("=============================================")
		else:
			pass
