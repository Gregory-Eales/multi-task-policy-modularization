import torch

from .actor import Actor
from .critic import Critic
from .router import Router
from .encoder import Encoder
from .decoder import Decoder


class MTPM(torch.nn.Module):

	def __init__(self):
		
		self.actor = Actor()
		self.critic = Critic() # needs to be a q function
		self.router = Router()
		self.encoder = Encoder()
		self.decoder = Decoder()

	def forward(self, x):

		"""
		z - latent space
		z_r - latent routes
		p - policy
		"""

		out = torch.Tensor(x).to(self.device)

		z = self.encoder(out)

		z_r = self.router(z)

		p = self.actor(z, r)