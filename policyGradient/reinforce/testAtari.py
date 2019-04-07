import gym


def main():
## testing atari platform 
	env = gym.make("Pong-v0")
	print("obs step : ",env.observation_space) 
	print("action step : ",env.action_space.n)

	env.reset()
	for _ in range(1000):
		nextState,reward,terminal,info = env.step(env.action_space.sample())
		env.render()

		if terminal:
			print(terminal)
	env.close()


if __name__ =="__main__":
	main()