import pandas as pd
import numpy as np



def save_data(env_names, seeds, rewards):
	
	for env in env_names:
		pass





def main():

	x = np.random.random([100, 6])
	print(["Seed: {}".format(i) for i in range(5)])
	data = pd.DataFrame(
		x,
		columns=["Steps"] + ["Seed: {}".format(i) for i in range(5)],
		)

	print(data.head())

	data.to_csv("rewards_lunar_landar")


if __name__ == "__main__":
	main()