import torch
import numpy as np


class Router(torch.nn.Module):


	def __init__(self):
		pass


	def forward(self, z):

		# takes in a latent encoding of the environment
		# encodes it into a learned task classification
		# could be trained using policy gradient methods
		# multinomial action output, continuous RL

		# input the terminal signal to switch output?
		pass