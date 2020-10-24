import torch
from torch import nn


class LinearModule(torch.nn.Module):

	def __init__(self, dim, n_layers):

		super(LinearModule, self).__init__()
	
		self.layers = torch.nn.ModuleList()

		for i in range(n_layers):
			self.layers.append(
				nn.Sequential(
					nn.Linear(dim, dim),
					nn.LeakyReLU()
					)
				)

	def forward(self, x):

		for layer in self.layers:
			x = layer(x)

		return x

class ModularizedLayer(torch.nn.Module):

	def __init__(self, num_modules, dim, n_layers):

		super(ModularizedLayer, self).__init__()
		
			self.layers = torch.nn.ModuleList()



class ModularizedAC(torch.nn.Module):

	def __init__(self, actor_lr, epsilon):

		super(ActorCritic, self).__init__()

		self.epsilon = epsilon
		self.define_network()
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
		self.to(self.device)

	def normalize(self, tensor):
		return (tensor - tensor.mean()) / ((torch.std(tensor))+1e-5)

	def define_network(self):
		self.relu = torch.nn.LeakyReLU()
		self.leaky_relu = torch.nn.LeakyReLU()
		self.sigmoid = torch.nn.Sigmoid()
		self.tanh = torch.nn.Tanh()
		self.softmax = torch.nn.Softmax(dim=-1)

		size = 256

		self.p1 = torch.nn.Linear(4, size)
		self.p2 = torch.nn.Linear(size, size)

		self.v1 = torch.nn.Linear(4, size)
		self.v2 = torch.nn.Linear(size, size)

		self.pi = torch.nn.Linear(size, 2)
		self.value = torch.nn.Linear(size, 1)


		self.critic_loss = torch.nn.MSELoss()


	def forward(self, x):


		out = torch.Tensor(x).float().to(self.device)
		"""
		This is where modularization goes
		"""
		module_outs = []



		"""

		"""
		p = self.pi(p)
		pi = self.softmax(p).to(torch.device('cpu:0'))

		v = self.value(v).to(torch.device('cpu:0'))

		return pi, v


def main():

	lin = LinearModule(10, 20)

	x = torch.ones(100, 10)

	print(lin(x).shape)

if __name__ == "__main__":
	main()