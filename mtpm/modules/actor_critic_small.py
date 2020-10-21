import torch

class ActorCritic(torch.nn.Module):

	def __init__(self, actor_lr, epsilon):

		super(ActorCritic, self).__init__()

		self.epsilon = epsilon
		self.define_network()
		#self.optimizer = torch.optim.Adam(params=self.parameters(), lr=actor_lr)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
		self.to(self.device)

	def normalize(self, tensor):
		return (tensor - tensor.mean()/(torch.std(tensor))+1e-3)

	def define_network(self):
		self.relu = torch.nn.LeakyReLU()
		self.leaky_relu = torch.nn.LeakyReLU()
		self.sigmoid = torch.nn.Sigmoid()
		self.tanh = torch.nn.Tanh()
		self.softmax = torch.nn.Softmax(dim=1)


		self.p1 = torch.nn.Linear(4, 64)
		self.p2 = torch.nn.Linear(64, 64)

		self.v1 = torch.nn.Linear(4, 64)
		self.v2 = torch.nn.Linear(64, 64)

		self.pi = torch.nn.Linear(64, 2)
		self.value = torch.nn.Linear(64, 1)


		self.critic_loss = torch.nn.MSELoss()


	def forward(self, x):


		out = torch.Tensor(x).float().to(self.device)

		#out = self.normalize(out)

		p = self.p1(out)
		p = self.relu(p)
		p = self.p2(p)
		p = self.relu(p)
		p = self.pi(p)
		pi = self.softmax(p).to(torch.device('cpu:0'))


		v = self.v1(out)
		v = self.relu(v)
		v = self.v2(v)
		v = self.relu(v)
		v = self.value(v).to(torch.device('cpu:0'))

		return pi, v