"""
__author__ : Kumar shubham
__date__   : 29-01-2019
__desc__   : following code will model the interaction in cartPole  game will try to learn the agent to model the interaction.
"""
from agent import Agent
import tensorflow as tf 
import numpy as np 
import gym
import random 

MinEpisode=500 ## filling replay buffer with random episode information to inatiate training
EPS = 0.99 ## initial prob of exploration
class CartPoleEnv(object):
	
	def __init__(self, logAdd="./log",batchSize = 32,maxIter=1000,maxTrEpisode =30000 ,learnStep =400,explDecayStep=1000):
		"""
		logAdd : address for saving all the log information
		batchSize : size of the batch for training
		maxIter   : maximum number of iteration after which an episode will be terminated
		learnStep : learn after taking given no of step
		explDecayStep : decay exploration after given step
		"""

		self.logAdd = logAdd ## by default ./log is taken as log add
		self.batchSize = batchSize
		self.maxIter = maxIter
		self.env = gym.make("CartPole-v1")
		self.obsSpace = self.env.observation_space.shape[0]
		self.actSpace = self.env.action_space.n
		self.explDecayStep = explDecayStep
		self.learnStep = learnStep
		self.maxTrEpisode = maxTrEpisode
		## creating an agent for processing
		self.agent = Agent(stateDim=self.obsSpace,noAction=self.actSpace,logAdd=self.logAdd,batchSize=self.batchSize)

		## initialize the memory of the agent
		self.initReplay() 

	def initReplay(self):
		## filling replayBuffer
		print ("initializing the memory... \n ")
		episode =0 
		while(episode <=MinEpisode):
			state = self.env.reset()
			i = 0
			## run for maximum given no of iteration
			while(i<self.maxIter):
				action = random.choice(np.arange(self.actSpace))
				nextState,reward,terminal,info = self.env.step(action)
				self.agent.replayBufMemory.add(currentState=state,action=action,reward=reward,nextState=nextState,done=terminal)
				if terminal:
					break
				state = nextState
				i+= 1
			episode+=1
		print ("replay buffer initialization done .. \n")
		print("length of the agent buffer : ",len(self.agent.replayBufMemory))

	def learnToPlay(self):
	## training of the system happens in given function
		print("======= STARTING LEARNING ======= \n")

		episode =0 
		exploreProb =EPS

		totalStep = 0
		while(episode <=self.maxTrEpisode):
			state = self.env.reset()
			i = 0
			## run for maximum given no of iteration
			while(i<self.maxIter):
				action = self.agent.action(currentState = state,exploreProb= exploreProb)
				nextState,reward,terminal,info = self.env.step(action)
				self.agent.replayBufMemory.add(currentState=state,action=action,reward=reward,nextState=nextState,done=terminal)

				if terminal:
					break
				state = nextState
				i+= 1
				totalStep+=1

				if (totalStep%self.learnStep == 0):
					## learning in given system
					cost = self.agent.learn()
					print("episode : {0} totalStep : {1} cost : {2}".format(episode,totalStep,cost))


				if (totalStep%self.explDecayStep ==0):
					exploreProb = exploreProb*EPS

			episode+=1
		############################
		print ("saving the model....")
		## Note : In current setting model is saved after whole trianing process end. It's not saving intermidate model per epoch
		self.agent.saveModel()


if __name__ == "__main__":
	obj = CartPoleEnv()
	state = obj.env.reset()
	obj.learnToPlay()
	# state_next, reward, terminal, info = obj.env.step(1)
	# print (state_next)
	# print(reward)
	# print (obj.obsSpace)
	# print(obj.actSpace)
