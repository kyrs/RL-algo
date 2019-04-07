"""
__author__ : Kumar shubham
__date__   : 27-Jan-2019
__desc__   : implementation of an agent for simple policy gradient implementation in pong environment
"""

import tensorflow as tf 
import numpy as np

class Monte_Carlo_Reinforce(object):
	def __init__(self,stateDim,noAction,activation,learningRate,summaryAdd,modelAdd):
		
		"""
		stateDim    :    dim of state feature in given environment
		noAction    :    no of valid action per state
		activation  :    activation defined per layer
		learningRate :   rate with which we take a step in gradient decent
		summaryAdd   :   add to save summary at
		modelAdd     :   add where model is saved
		"""
		self.stateDim = stateDim 
		self.noAction = noAction
		self.activation = activation
		self.learningRate = learningRate
		self.summaryAdd = summaryAdd
		self.modelAdd = modelAdd
		
	
	def preprocessImage(self,image):
		## given function does the preprocess of the image before passing it for prediction 

		## following function convert 210x160x3 uint8 frame into 6400 (80x80) 1D float vector 

		Image = image[35:195]
		Image = Image[::2,::2,0] ## downsampling of the image
		Image[Image == 144] = 0 ## removing one background
		Image[Image == 109] = 0 ## removing one background
		Image[Image!=0]=1 ## set rest as 1 
		finalImage = Image.astype(np.float).ravel()
		return finalImage


	def buildNetwork(self):
		## defining  and building the network architecture


		self.input = tf.placeholder(tf.float32,shape=[None,self.stateDim],name= "input")
		self.action = tf.placeholder(tf.int32,shape=[None],name= "action")
		self.actionTaken = tf.one_hot(self.action, self.noAction,name="actionTaken")
		self.valueReward = tf.placeholder(tf.float32,shape=[None],name= "discountReward") ## used for policy gradient 

		self.meanReward_ = tf.placeholder(tf.float32,name = "meanreward")## to put it on tensorboard

		self.lossTf_ = tf.placeholder(tf.float32,name = "LossTfboard")
		self.fc1 = tf.layers.dense(self.input,64,activation=self.activation, name = "fc1")
		self.fc2 = tf.layers.dense(self.fc1,32,activation=self.activation, name = "fc2")
		self.logit = tf.layers.dense(self.fc2,self.noAction,name="logit")

		## action distribution of the system
		self.actionDist = tf.nn.softmax(self.logit)

		self.negLogProb = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.logit, labels = self.actionTaken)
		self.loss = tf.reduce_mean(self.negLogProb*self.valueReward)
		self.opt = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)



		self.summary() ## building the summary part also for the network

	def summary(self):
		## merging the summary operation 
		with tf.name_scope("summary"):
			tf.summary.scalar('totalReward',self.meanReward_)
			tf.summary.scalar('Loss',self.lossTf_)
			self.summaryOp = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter(self.summaryAdd, tf.get_default_graph())

	def learnNetwork(self,inputState,valueReward,action,sess):
		## training the system 

		"""
		inputState : batch of state information
		valueReward : value function for each state 
		action     : action taken to reach a given state
		sess 	   : sess for running the train process


	
		OUTPUT : 
		lossValue : cummulative loss over the batch 
		
		Note:
		make sure network and summary is build before running given part

		""" 
		lossValue,_ = sess.run([self.loss,self.opt], {self.input :inputState,self.action:action,self.valueReward:valueReward})
		return np.mean(lossValue)

	def writeAvgScore(self,meanreward,loss,sess):
		## saving the score value 
		summary = sess.run(self.summaryOp,{self.meanReward_:meanreward,self.lossTf_:loss})
		self.writer.add_summary(summary)

	def forwardPass(self,inputState,sess):
		## return the possible action for given state 

		"""
		inputState : state for which prediciton about possible action need to be made
		sess       : sess for running the prediction
		
		output : 

		action : possible action for given state

		""" 
		options = np.arange(self.noAction)
		actionDist = sess.run([self.actionDist],{self.input:inputState})
		return np.random.choice(options,p=actionDist[0].ravel()) ## dirichlet distribution over softmax

	
	def valueFnGen(self,listOfReward,discount):
		## genereate valuefunction per state for given distribution of policy 
		cummValueList = np.zeros(len(listOfReward))

		cummulative=0

		for i in range(len(listOfReward)-1,-1,-1):
			cummulative = cummulative*self.discount+listOfReward[i]
			cummValueList[i] = cummulative

		mean = np.mean(cummValueList)
		std = np.std(cummValueList)

		return (cummValueList-mean)/(std)

