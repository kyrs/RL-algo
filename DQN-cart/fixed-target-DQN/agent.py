"""
__author__ : Kumar shubham 
__date__   : 18-02-2019
__desc__   : code for agent 
"""
import numpy as np 
import tensorflow as tf 
from collections import deque,namedtuple
import random 
from model import Model
import os
from tensorflow.python.framework import ops
from copy import deepcopy

class FixedTargetAgent:
	def __init__(self,stateDim,noAction,logAdd,batchSize,train=True,seed = 100, gamma = 0.8,explDecayRate=0.99,gpuPercent=0.25):
		"""
		stateDim    :    dim of state feature in given environment
		noAction    :    no of valid action per state
		logAdd      :    add to save all the log files
		seed 		:    randomness seed 
		train       :    flag for train / infer
		batchSize   :    batch to consider for training
		gamma       :    how much you wanna focus on future in Q learning
		explDecayRate :  rate with which exploration is going to be decayed down
		gpuPercent : 	percentage of gpu used by per network
		"""

		self.stateDim = stateDim
		self.noAction = noAction
		self.logAdd   = logAdd
		self.batchSize = batchSize
		self.explDecay = explDecayRate
		self.seed = random.seed(seed)
		self.gamma = gamma

		self.trainFlag = train
		self.replayBufMemory = replayBuffer(maxLen=10000,seed = 100,batchSize = self.batchSize)

		summAdd = os.path.join(self.logAdd,"summary")
		modelAdd = os.path.join(self.logAdd,"model")

		self.targetNet = Model(self.stateDim,self.noAction,tf.nn.relu,0.001,summAdd,modelAdd)
		self.actualNet = Model(self.stateDim,self.noAction,tf.nn.relu,0.001,summAdd,modelAdd)


		gpuOptions = tf.GPUOptions(per_process_gpu_memory_fraction=gpuPercent)
		config=tf.ConfigProto(gpu_options=gpuOptions)
		## building the network
		self.targetNet.buildNetwork(config = config)
		self.actualNet.buildNetwork(config = config)

		

		## restore the save ckpt files 
		if not self.trainFlag:
			self.restoreCkpt()
		
		with self.targetNet.graph.as_default(): 
			self.targetNet.sess.run(tf.global_variables_initializer())

		with self.actualNet.graph.as_default():
			self.actualNet.sess.run(tf.global_variables_initializer())

	def learn(self):
		## code for trainig the network

		batchCurrentState,batchAction,batchReward,batchNextState,batchDone = self.replayBufMemory.sample()

		expectedQValue = []
		currStateInfo = []
		actionForPred = []
		
		## predicting the probability for the next states 
		predQvalNextState = self.targetNet.forwardPass(inputState=batchNextState) 

		for currState,currAction,currReward,nextState,ifDone,qvalNextState in zip(batchCurrentState,batchAction,batchReward,batchNextState,batchDone,predQvalNextState[0]):
			currStateInfo.append(currState)
			## coverting action to one ht encoding
			temp = np.zeros((1,self.noAction))
			temp[0,currAction[0]]=1
			actionForPred.append(temp)
			expectedQValue.append(currReward[0] + self.gamma *np.max(qvalNextState) *(1-ifDone))## formulation for Q value
			

		stateMat = np.vstack(currStateInfo)
		qValMat = np.vstack(expectedQValue)
		actMat = np.vstack(actionForPred)

		## train the network 
		cost = self.actualNet.learnNetwork(inputState=stateMat,targetQ=qValMat,action=actMat)
		return cost


	def action(self, currentState, exploreProb):
		## return the action for current state
		"""
		currentState : current state feataure vector for making the prediction
		exploreProb  : probability of exploration
		RETURN :
		possible action based on exploration probability
		"""
		val = random.random()
		if (val>exploreProb):
			output = np.array(currentState).reshape(1,self.stateDim)
			actionProb = self.actualNet.forwardPass(inputState=output)
			return np.argmax(actionProb)
		else:
			return random.choice(np.arange(self.noAction)) 

	def saveModel(self):
		## save the model
		saver=tf.train.Saver()
		saver.save(self.actualNet.sess, self.modelAdd) 

	def restoreCkpt(self):
		#restore the model in given path
		print ("model restored")
		saver=tf.train.Saver()
		saver.restore(self.actualNet.sess, self.modelAdd)

	def tranfActToTar(self):
		## method to transfer the weights of actual network to target network
		# Get the parameters of our DQNNetwork
		fromVars = self.actualNet.sess.run(self.actualNet.varList)
		
		## get the parameters of the target DNN
		toVars = self.targetNet.varList

		opHolder = []
		
		# Update our target_network parameters with DQNNetwork parameters (POSSIBILITY OF MEMORY LEAK  NOTE : FIND AN ALTERNATIVE using PLACEHOLDER)
		with self.targetNet.graph.as_default():
			for fromVar,toVar in zip(fromVars,toVars):
				opHolder.append(toVar.assign(fromVar))

		self.targetNet.sess.run(opHolder)		

		
class replayBuffer(object):
	def __init__(self,maxLen,seed,batchSize):
		"""
		maxLen : max length of the deque
		seed   : defining the random state
		batchSize : batchsize for sampling  
		"""
		self.maxLen = maxLen
		self.seed = random.seed(seed)
		self.batchSize = batchSize
		self.memory = deque(maxlen = self.maxLen)

		self.experience = namedtuple("experience", ["currentState","action","reward","nextState","done"])

	def add(self,currentState,action,reward,nextState,done):
		## items to add into the deque 
		"""
		currentState : state where system is currently in 
		action 		 : action taken by the system in current state
		reward 		 : reward got by agent on taking given action
		nextState    : next state which agent moved to after taking given action 
		done 		 : flag pointing whether it has reached final state or not

		"""
		exp = self.experience(currentState,action,reward,nextState,done)
		self.memory.append(exp)

	def sample(self):
		## method for sampling the data from the deque

		## OUTPUT :  vstacked sampled output
		batch = random.sample(self.memory,self.batchSize)

		batchCurrentState= np.vstack([exp.currentState for exp in batch if exp is not None])
		batchAction 	 = np.vstack([exp.action for exp in batch if exp is not None])
		batchReward 	 = np.vstack([exp.reward for exp in batch if exp is not None])
		batchNextState	 = np.vstack([exp.nextState for exp in batch if exp is not None])
		batchDone 		 = np.vstack([exp.done for exp in batch if exp is not None])

		return (batchCurrentState,batchAction,batchReward,batchNextState,batchDone)

	def __len__(self):
		## defining the length option for the agent
		return len(self.memory)

	

