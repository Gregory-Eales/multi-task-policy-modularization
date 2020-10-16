from matplotlib import pyplot as plt
import numpy as np
import random


def generate_graphs(agent, path=None):

	reward_per_epoch(agent.get_rewards(), path=path)



def plot_rewards(rewards, path=None):

	rewards = np.array([np.array(r) for r in rewards])


	print(rewards.shape)

	mean_rewards = rewards.mean(axis=0)

	print(mean_rewards.shape)

	plt.plot(mean_rewards, label="mean rewards")

	plt.fill_between(
		list(range(rewards.shape[1])),
		np.amax(rewards, axis=0),
		np.amin(rewards, axis=0),
		alpha=0.2)

	plt.title("Reward per Epoch")
	plt.xlabel("Epoch")
	plt.ylabel("Reward")
	plt.legend(loc="upper left")


	if path == None:
		plt.show()

	else:
		plt.savefig('{}/graphs/reward_per_update.png'.format(path))


def reward_per_epoch(rewards, path=None):

	plt.title("Reward per Epoch")
	plt.xlabel("Epoch")
	plt.ylabel("Reward")
	plt.plot(reward, label="reward")
	plt.legend(loc="upper left")


	if path == None:
		plt.show()

	else:
		plt.savefig('{}/graphs/reward_per_epoch.png'.format(path))


	
	# CONFIDENCE BANDS !
	
	N = 21
	x = np.linspace(0, 10, 11)
	y = [3.9, 4.4, 10.8, 10.3, 11.2, 13.1, 14.1,  9.9, 13.9, 15.1, 12.5]

	# fit a linear curve an estimate its y-values and their error.
	a, b = np.polyfit(x, y, deg=1)
	y_est = a * x + b
	y_err = x.std() * np.sqrt(1/len(x) +
	                          (x - x.mean())**2 / np.sum((x - x.mean())**2))

	fig, ax = plt.subplots()
	ax.plot(x, y_est, '-')
	ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
	ax.plot(x, y, 'o', color='tab:brown')
	
	plt.show()

def main():

	paths = []
	for j in range(5):
		path = [0]
		for i in range(1, 500):
			path.append(path[i-1]+random.random())
		paths.append(path)

	num = np.sqrt(np.array(paths).reshape([500, 5]))#+np.linspace(1, 500, num=500).reshape([500, 1])**0.5
	#reward_per_epoch(num, path=None)

	num_avg = np.sum(num, axis=1)/5
	print(num[num.argmax(axis=1)].shape)
	print(num.argmin(axis=1))
	plt.plot(num_avg)
	plt.plot(num)
	plt.fill_between(list(range(500)),np.amax(num, axis=1), np.amin(num, axis=1), alpha=0.2)
	#plt.plot(np.amax(num, axis=1))
	#plt.plot(np.amin(num, axis=1))
	plt.show()

if __name__ == "__main__":
	main()