"""
__author__ : Kumar shubham
__date__   : 29-01-2019
__desc__   : following code will model the interaction in cartPole  game will try to learn the agent to model the interaction.


NOTE : xming setting set DISPLAY=:0
"""
from agent import Agent
import tensorflow as tf 
import numpy as np 
import gym
import random 


TRAIN = False

class gymEnv_pingpong(object):
	
	def __init__(self, logAdd="./log",batchSize = 32,maxIter=5000,testEpisode = 50,trainIter=100):
		"""
		logAdd : address for saving all the log information
		batchSize : size of the batch for training
		maxIter   : maximum number of iteration after which an episode will be terminated
		explDecayStep : decay exploration after given step
		testEpisode : no of time to show the test
		trainIter : no of episode to be run for training
		"""

		self.logAdd = logAdd ## by default ./log is taken as log add
		self.batchSize = batchSize
		self.maxIter = maxIter
		self.env = gym.make("Pong-v0")
		self.actSpace = self.env.action_space.n
		self.maxTrEpisode = trainIter
		self.testEpisode = testEpisode
		## creating an agent for processing
		self.agent = Agent(noAction=self.actSpace,logAdd=self.logAdd,batchSize=self.batchSize,train=TRAIN)

		

	def learnToPlay(self):
	## training of the system happens in given function
		print("======= STARTING LEARNING ======= \n")

		episode =0 
		
		totalStep = 0
		episodicRewardList = []
		while(episode <=self.maxTrEpisode):
			state = self.env.reset()
			i = 0
			## run for maximum given no of iteration
			stateList = []
			rewardList = []
			actionList = []
			prevState = None
			while(i<self.maxIter):
				if (episode ==0):
					action = self.env.action_space.sample()
					nextState,reward,terminal,info = self.env.step(action)

					rewardList.append(reward)## appending the reward
					stateList.append(nextState) ## appending the state
					actionList.append(action) ## appending the action

					if terminal:
						break

				else:
					action = self.agent.Action(currentState=state,prevState=prevState)
					newState,reward,terminal,info = self.env.step(action)
					prevState=state
					state = newState	


					rewardList.append(reward)## appending the reward
					stateList.append(nextState) ## appending the state
					actionList.append(action) ## appending the action

					if terminal:
						break				


			episode+=1
			episodicRewardList.append(np.mean(rewardList))
				## learning per episode 
				
			

			loss = self.agent.learn(stateList=stateList,rewardList = rewardList,actionList = actionList)
			self.agent.writeAvgScore(meanreward=np.mean(episodicRewardList),loss=loss,sess=self.agent.sess)

			print ("episode: %d loss : %f reward : %f"%(episode, loss, np.mean(episodicRewardList)))

		############################
		print ("saving the model....")
		## Note : In current setting model is saved after whole trianing process end. It's not saving intermidate model per epoch
		self.agent.saveModel()


	def play(self):
		## play the game
		episode =0 
		
		totalStep = 0
		while(episode <=self.testEpisode):
			state = self.env.reset()
			prevState = None
			i = 0
			## run for maximum given no of iteration
			while(i<self.maxIter):
				action = self.agent.Action(currentState=state,prevState=prevState)
				newState,reward,terminal,info = self.env.step(action)
				prevState=state
				state = newState
				self.env.render()

				if terminal:
					break

				state = newState
				i+=1
			episode+=1


if __name__ == "__main__":
	obj = gymEnv_pingpong()
	# uncomment it to learn
	# obj.learnToPlay()

	## uncomment it to play
	obj.play()
	