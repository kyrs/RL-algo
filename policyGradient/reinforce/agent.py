"""
__author__ : Kumar shubham 
__date__   : 28-01-2019
__desc__   : code for agent 
"""
import numpy as np 
import tensorflow as tf 
# from collections import deque,namedtuple
import random 
from model import Monte_Carlo_Reinforce
import os

### for handling image : we have used the concept mentioned by Karpathy in his gist to process image and take diff for action
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

## 80*80 input dims is based on down sampling and internal pre processing .. which is srt of hard coded
class Agent(Monte_Carlo_Reinforce):
	def __init__(self,noAction,logAdd,batchSize,inputDim=80*80,train=True,seed = 100, gamma = 0.8):
		"""
		inputDim    :    iputdim of the model
		noAction    :    no of valid action per state
		logAdd      :    add to save all the log files
		seed 		:    randomness seed 
		train       :    flag for train / infer
		batchSize   :    batch to consider for training
		gamma       :    how much you wanna focus on future 
		"""

		self.inputDim = inputDim
		self.noAction = noAction
		self.logAdd   = logAdd
		self.seed = random.seed(seed)
		self.gamma = gamma
		self.batchSize=batchSize
		self.trainFlag = train

		summAdd = os.path.join(self.logAdd,"summary")
		modelAdd = os.path.join(self.logAdd,"model")

		if os.path.isdir(modelAdd):
			pass
		else:
			os.mkdir(modelAdd)

		if os.path.isdir(summAdd):
			pass
		else:
			os.mkdir(summAdd)

		modelAdd = modelAdd+"/reinforceModel"
		summAdd = summAdd+"/summaryWriter"
		super(Agent, self).__init__(self.inputDim,self.noAction,tf.nn.relu,0.001,summAdd,modelAdd)

		## building the network
		self.buildNetwork()

		self.sess = tf.Session()

		## restore the save ckpt files 
		if not self.trainFlag:
			self.restoreCkpt()
		else:
			self.sess.run(tf.global_variables_initializer())

	def learn(self,stateList=[],rewardList = [],actionList = []):
		imageList = []

		imageList.append(np.zeros((1,self.inputDim)))
		for frameIdx in range(1,len(stateList)):
			prevImage = stateList[frameIdx-1]
			origImage = stateList[frameIdx]

			## process PrevImage 
			proPrevImg  = self.preprocessImage(prevImage)
			proOrgImg = self.preprocessImage(origImage)

			diffImage = proOrgImg-proPrevImg 
			diffImage = diffImage[np.newaxis,...]
			imageList.append(diffImage)

		proFrameList = np.concatenate(imageList,axis=0) ## frameList for training
		valueFnList = self.valueFnGen(rewardList) ## value function

		lossMean = []
		
		
		for idx in range(0,len(imageList),self.batchSize):

			out = self.learnNetwork(inputState=proFrameList[idx:idx+self.batchSize],valueReward=rewardList[idx:idx+self.batchSize],action=actionList[idx:idx+self.batchSize],sess=self.sess)
			print("output loss : ",out)
			lossMean.append(out)
		return np.mean(lossMean)

	def valueFnGen(self,listOfReward):
		## genereate valuefunction per state for given distribution of policy 
		cummValueList = np.zeros(len(listOfReward))

		cummulative=0

		for i in range(len(listOfReward)-1,-1,-1):
			cummulative = cummulative*self.gamma+listOfReward[i]
			cummValueList[i] = cummulative

		mean = np.mean(cummValueList)
		std = np.std(cummValueList)

		return (cummValueList-mean)/(std)
		
	def Action(self, currentState,prevState):
		## return the action for current state
		if prevState is None:
			diffState = np.zeros((1,self.inputDim))
		else:
			proPrevImg  = self.preprocessImage(prevState)
			proOrgImg = self.preprocessImage(currentState)

			diffState = proOrgImg-proPrevImg

			diffState = diffState[np.newaxis,...]
		action = self.forwardPass(diffState,self.sess)
		return action

		
	def saveModel(self):
		## save the model
		saver=tf.train.Saver()
		saver.save(self.sess, self.modelAdd) 

	def restoreCkpt(self):
		#restore the model in given path
		print ("model restored")
		saver=tf.train.Saver()
		saver.restore(self.sess, self.modelAdd)
		
