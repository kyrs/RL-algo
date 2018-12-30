"""
__author__ : kumar shubham
__date__   : 25-12-2018
__desc__   : file for creating the interaction of agent
"""
import numpy as np 
import gym 

def mainInt(env,agent):

	noEpisode =75000 ## episode to run
	valueFnUpd = 400 ## after how many episode Q function is printed
	maxStepPrEps = 300 ## max step per episode

	for epCnt in range(1,noEpisode):
		observation  = env.reset()
		episodeList = list()
		for i in range(maxStepPrEps):
			## taking an action
			action = agent.action(observation,1/epCnt)
			## taking the needed action
			observation,reward,done,info = env.step(action)  
			episodeList.append((observation,action,reward))
			if done:
				## episode reached
				break

		agent.episodicUpdate(episodeList)
		agent.updateQFn() ## update Q function after every episode 

		if (epCnt% valueFnUpd==0):
			
			env.render()
			# print ("======Value fn after : %d epoch cnt======== "%(epCnt))
			
			# print("=============================================")
		else:
			pass
	print(agent.QFunction)