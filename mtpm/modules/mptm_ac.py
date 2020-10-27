import torch
from torch import nn


class LinearModule(torch.nn.Module):

	def __init__(self, in_dim, h_dim, n_layers):

		super(LinearModule, self).__init__()
	
		self.layers = torch.nn.ModuleList()

		self.layers.append(
				nn.Sequential(
					nn.Linear(in_dim, h_dim),
					nn.LeakyReLU()
					)
				)

		for i in range(n_layers-1):
			self.layers.append(
				nn.Sequential(
					nn.Linear(h_dim, h_dim),
					nn.LeakyReLU()
					)
				)

	def forward(self, x):

		for layer in self.layers:
			x = layer(x)

		return x

class ModularizedLayer(torch.nn.Module):

	def __init__(self, n_modules, in_dim, h_dim, n_layers):

		super(ModularizedLayer, self).__init__()
		
		self.mod_layers = torch.nn.ModuleList()


		for i in range(n_modules):
			self.mod_layers.append(LinearModule(in_dim, h_dim, n_layers))

	def forward(self, x):

		outs = []

		for m in self.mod_layers:
			outs.append(m(x))

		return torch.cat(outs, dim=1)


class ModularizedAC(torch.nn.Module):

	def __init__(self, actor_lr, epsilon):

		super(ModularizedAC, self).__init__()

		self.epsilon = epsilon
		self.define_network()
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
		self.to(self.device)


	def gradient_clip_hook(self, grad, clip_value):
		for p in self.parameters():
    		p.register_hook(
    			lambda grad: torch.clamp(grad, -clip_value, clip_value)
    			)

	def normalize(self, tensor):
		return (tensor - tensor.mean()) / ((torch.std(tensor))+1e-5)

	def define_network(self):
		self.relu = torch.nn.LeakyReLU()
		self.leaky_relu = torch.nn.LeakyReLU()
		self.sigmoid = torch.nn.Sigmoid()
		self.tanh = torch.nn.Tanh()
		self.softmax = torch.nn.Softmax(dim=-1)

		size = 128

		self.m1 = ModularizedLayer(4, 8, 32, 2)
		self.m2 = ModularizedLayer(4, 128, 32, 2)

		self.v1 = torch.nn.Linear(size, size)
		self.p1 = torch.nn.Linear(size, size)
			

		self.pi = torch.nn.Linear(size, 2)
		self.value = torch.nn.Linear(size, 1)


		self.critic_loss = torch.nn.MSELoss()


	def forward(self, x):


		out = torch.Tensor(x).float().to(self.device)
		"""
		This is where modularization goes
		"""
		out = self.m1(out)
		out = self.m2(out)

		"""

		"""
		p = self.p1(out)
		p = self.pi(p)
		pi = self.softmax(p).to(torch.device('cpu:0'))

		v = self.v1(out)
		v = self.value(v).to(torch.device('cpu:0'))

		return pi, v


def main():

	l1 = LinearModule(5, 1, 2)
	l2 = LinearModule(5, 1, 2)
	l3 = LinearModule(5, 1, 2)
	l4 = LinearModule(5, 1, 2)
	l5 = LinearModule(5, 1, 2)

	x = torch.ones(10, 5)
	print(l1(x).shape)

	
	out = torch.cat(
		[
		l1(x),
		l2(x),
		l3(x),
		l4(x),
		l5(x),
		],
		dim=1
		)

	
	print(out.shape)

if __name__ == "__main__":
	main()