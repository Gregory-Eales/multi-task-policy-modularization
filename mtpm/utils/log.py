import logging

def init_log(path):
	logging.basicConfig(filename='example.log', level=logging.DEBUG)

def log():

	#filename='example.log'

	"""
	logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
	logging.warning('This will get logged to a file')
	"""

	logging.basicConfig(level=logging.DEBUG, format='%(asctime)s  %(message)s', datefmt='[%d-%b-%y %H:%M:%S]')

	logging.debug('This is a debug message')
	logging.info('This is an info message')
	logging.warning('This is a warning message')
	logging.error('This is an error message')
	logging.critical('This is a critical message')

	#print("Average Episode Length: {}".format(
	   # np.sum(episode_lengths)/len(episode_lengths)))

	#print("Episode finished after {} timesteps".format(step+1))
	#print("Largest Episode Length: {}".format(max(episode_lengths)))

	#print("Epoch: {}".format(epoch))

if __name__ == "__main__":
	log()