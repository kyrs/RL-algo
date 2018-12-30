"""
__name__ : Kumar Shubham
__Date__ : 25-12-2018
__desc__ : file to setup the gym interface
"""
import gym 
import numpy as np
from interaction_SARSA import mainInt
from agent_SARSA import AgentPrediction_SARSA

def main():
	env = gym.make("Taxi-v2")
	actionSpace = env.action_space.n 
	obsSpace = env.observation_space.n
	
	## defining the agent 
	agent = AgentPrediction_SARSA(gamma=0.7,noActSpace=actionSpace,noObsSpace=obsSpace,alpha =0.3)

	## interaction with the environment
	mainInt(env,agent)


if __name__ == "__main__":
	main()