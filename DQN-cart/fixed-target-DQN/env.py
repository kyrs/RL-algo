"""
__author__ : Kumar shubham
__date__   : 18-02-2019
__desc__   : following code will model the interaction in cartPole  game will try to learn the agent to model the interaction.
"""
from agent import FixedTargetAgent as Agent
import tensorflow as tf 
import numpy as np 
import gym
import random 

MinEpisode=500 ## filling replay buffer with random episode information to inatiate training
EPS = 0.99 ## initial prob of exploration

TRAIN = False

class gymEnv(object):
	
	def __init__(self, logAdd="./log",batchSize = 32,maxIter=5000,maxTrEpisode =9500 ,testEpisode = 50,learnStep =400,explDecayStep=25000,switchTargStep = 5000):
		"""
		logAdd : address for saving all the log information
		batchSize : size of the batch for training
		maxIter   : maximum number of iteration after which an episode will be terminated
		learnStep : learn after taking given no of step
		explDecayStep : decay exploration after given step
		testEpisode : no of time to show the test
		switchTargStep : steps after which the target and actual network is switched
		"""

		self.logAdd = logAdd ## by default ./log is taken as log add
		self.batchSize = batchSize
		self.maxIter = maxIter
		self.env = gym.make("Acrobot-v1")
		self.obsSpace = self.env.observation_space.shape[0]
		self.actSpace = self.env.action_space.n
		self.explDecayStep = explDecayStep
		self.learnStep = learnStep
		self.maxTrEpisode = maxTrEpisode
		self.testEpisode = testEpisode
		## creating an agent for processing
		self.agent = Agent(stateDim=self.obsSpace,noAction=self.actSpace,logAdd=self.logAdd,batchSize=self.batchSize,train=TRAIN)
		self.negativeReward = -0.0001
		## initialize the memory of the agent
		if (TRAIN):
			self.initReplay() 

		self.switchTargStep = switchTargStep

		

	def initReplay(self):
		## filling replayBuffer
		print ("initializing the memory... \n ")
		episode =0 
		while(episode <=MinEpisode):
			state = self.env.reset()
			i = 0
			## run for maximum given no of iteration
			####################################################################################################
			# during testing it took too much time for model to respond
			# so to fasten things up we are adding a negative reward of -0.0001 per step 
			######################################################################################

			while(i<self.maxIter):
				action = random.choice(np.arange(self.actSpace))
				nextState,reward,terminal,info = self.env.step(action)
				reward += self.negativeReward*i ## negative reward
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
				reward += self.negativeReward*i ## negative reward
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


				if (totalStep%self.switchTargStep==0):
					print("copying parameters...")
					self.agent.tranfActToTar()

				if (totalStep%self.explDecayStep ==0):
					exploreProb = exploreProb*EPS

			episode+=1
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
			i = 0
			## run for maximum given no of iteration
			while(i<self.maxIter):
				action = self.agent.action(currentState = state,exploreProb= 0.0)
				nextState,reward,terminal,info = self.env.step(action)
				self.env.render()

				if terminal:
					break

				state = nextState
				i+=1
			episode+=1


if __name__ == "__main__":
	obj = gymEnv()
	

	if (TRAIN):
	# uncomment it to learn
		obj.learnToPlay()
	else:
	## uncomment it to play
		obj.play()

	