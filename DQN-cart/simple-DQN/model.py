"""
__author__ : Kumar shubham
__date__   : 27-Jan-2019
__desc__   : implementation of an agent for simple DQN implementation
"""

import tensorflow as tf 
import numpy as np

class Model(object):
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
		
	def buildNetwork(self):
		## defining  and building the network architecture


		self.input = tf.placeholder(tf.float32,shape=[None,self.stateDim],name= "input")
		self.actionTaken = tf.placeholder(tf.float32,shape=[None,self.noAction],name= "actionTaken")
		self.targetQvalue = tf.placeholder(tf.float32,shape=[None,1],name= "targetQ")

		self.fc1 = tf.layers.dense(self.input,64,activation=self.activation, name = "fc1")
		self.logit = tf.layers.dense(self.fc1,self.noAction, name = "logit")

		with tf.name_scope("loss-prediction"):
			self.predictedQ = tf.reduce_sum(tf.multiply(self.logit,self.actionTaken))
			self.loss = tf.reduce_mean(tf.square(self.predictedQ-self.targetQvalue))
			self.optimier = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)

		self.summary() ## building the summary part also for the network

	def summary(self):
		## merging the summary operation 
		with tf.name_scope("summary"):
			tf.summary.scalar('loss',self.loss)
			self.summaryOp = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter(self.summaryAdd, tf.get_default_graph())

	def learnNetwork(self,inputState,targetQ,action,sess):
		## training the system 

		"""
		inputState : batch of state information
		targetQ    : target value of Q function 
		action     : action taken to reach a given state
		sess 	   : sess for running the train process

	
		OUTPUT : 
		lossValue : cummulative loss over the batch 
		
		Note:
		make sure network and summary is build before running given part

		""" 
		lossValue,summary,_ = sess.run([self.loss,self.summaryOp,self.optimier], {self.input :inputState,self.actionTaken:action,self.targetQvalue:targetQ})
		self.writer.add_summary(summary)
		return np.mean(lossValue)


	def forwardPass(self,inputState,sess):
		## return the possible action for given state 

		"""
		inputState : state for which prediciton about possible action need to be made
		sess       : sess for running the prediction
		
		output : 

		Qsa  : possible action for given state 

		""" 
		Qsa = sess.run([self.logit],{self.input:inputState})
		return Qsa

	def saveModel(self,sess):
		## save the model
		saver=tf.train.Saver()
		saver.save(sess, self.modelAdd) 

