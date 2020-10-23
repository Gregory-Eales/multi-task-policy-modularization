import pandas as pd
import numpy as np
import pickle
import json



def save_arguments(path="", args=None):

	if args!=None:
		file = open("{}/arguments.json".format(path), "w", encoding="utf8")
		json.dump(vars(args), file)
		file.close()

def load_arguments(path=""):
	file = open("{}arguments.json".format(path), "rb")
	return json.load(file)


def save_results(env_name, seeds, rewards, update_episodes, path=""):

	rewards = np.array([np.array(r) for r in rewards]).T


	seeds = ["Seed: {}".format(seed) for seed in seeds]

	data = pd.DataFrame(rewards, columns=seeds)

	data["Episodes"] = update_episodes*np.array(list(range(rewards.shape[0])))

	cols = data.columns.tolist()
	cols = cols[-1:] + cols[:-1]
	data = data[cols]
	data.index.name = "Index"
	print(data.head())

	if path != "":
		data.to_csv("{}/results/{}_rewards.csv".format(path, env_name))

	else:
		data.to_csv("{}_rewards.csv".format(env_name))

def main():

	from argparse import ArgumentParser

	parser = ArgumentParser(add_help=False)

	# experiment and  environment
	parser.add_argument('--experiment_name', default="default", type=str)
	parser.add_argument(
	'--env_names',
	default=["Breakout-ram-v0"]#,"Pong-ram-v0"]
	)
	args = parser.parse_args()

	print(vars(args))


if __name__ == "__main__":
	main()