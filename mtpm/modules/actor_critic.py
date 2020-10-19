import torch
from tqdm import tqdm
import numpy as np


class ResBlock(torch.nn.Module):

	def __init__(self, num_channel):

		super(ResBlock, self).__init__()

		self.conv1 = torch.nn.Conv2d(
			num_channel,
			num_channel,
			kernel_size=3,
			stride=1,
			padding=1
			)

		self.conv2 = torch.nn.Conv2d(
			num_channel,
			num_channel,
			kernel_size=3,
			stride=1,
			padding=1
			)

		self.leaky_relu = torch.nn.LeakyReLU()


	def forward(self, x):

		out = x

		out = self.leaky_relu(out)
		out = self.conv1(out)
		out = self.leaky_relu(out)
		out = self.conv2(out)
		
		return out + x



class ConvBlock(torch.nn.Module):

	def __init__(self, in_channel, num_channel):

		super(ConvBlock, self).__init__()

		self.conv = torch.nn.Conv2d(
			in_channel,
			num_channel,
			kernel_size=3,
			stride=1,
			)

		self.max = torch.nn.MaxPool2d(
			kernel_size=3,
			stride=2,
			)


		self.res1 = ResBlock(num_channel=num_channel)
		self.res2 = ResBlock(num_channel=num_channel)

		self.leaky_relu = torch.nn.LeakyReLU()


	def forward(self, x):

		out = x

		out = self.conv(out)
		out = self.max(out)
		out = self.res1(out)
		out = self.res2(out)

		return out



class ActorCritic(torch.nn.Module):

	def __init__(self, actor_lr, epsilon):

		super(ActorCritic, self).__init__()

		self.epsilon = epsilon
		self.define_network()
		self.optimizer = torch.optim.Adam(params=self.parameters(), lr=actor_lr)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
		self.to(self.device)
		self.prev_params = self.parameters()

	def define_network(self):
		self.relu = torch.nn.ReLU()
		self.leaky_relu = torch.nn.LeakyReLU()
		self.sigmoid = torch.nn.Sigmoid()
		self.tanh = torch.nn.Tanh()
		self.softmax = torch.nn.Softmax(dim=0)
		
		"""
		self.l1 = torch.nn.Linear(1024, 512)
		self.l2 = torch.nn.Linear(512, 64)
		self.l3 = torch.nn.Linear(64, 15)
		self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=4, stride=2)
		self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2)
		self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2)
		"""

		self.l1 = torch.nn.Linear(32*5*5, 512)
		self.l2 = torch.nn.Linear(512, 64)
		self.l3 = torch.nn.Linear(64, 15)
		
		self.block1 = ConvBlock(3, 16)
		self.block2 = ConvBlock(16, 32)
		self.block3 = ConvBlock(32, 32)

		self.critic_loss = torch.nn.MSELoss()

	def actor_loss(self, log_probs, k_log_probs, advantages):

		r_theta = torch.exp(log_probs-k_log_probs)

		clipped_r = torch.clamp(
			r_theta,
			1.0 - self.epsilon,
			1.0 + self.epsilon
			)

		return torch.mean(torch.min(r_theta*advantages, clipped_r*advantages))

	def forward(self, x):

		out = torch.Tensor(x).float().to(self.device)/255

		out = self.block1(out)
		out = self.block2(out)
		out = self.block3(out)
		out = self.leaky_relu(out)

		out = out.reshape(out.shape[0], 5*5*32)

		out = self.l1(out)
		out = self.leaky_relu(out)
		out = self.l2(out)
		out = self.leaky_relu(out)
		out = self.l3(out)

		out = self.softmax(out)

		return out.to(torch.device('cpu:0'))

	def optimize(
		self,
		log_probs,
		k_log_probs,
		advantages,
		batch_sz=32
		):

		loss = self.loss(log_probs, k_log_probs, advantages)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()


def main():


	block = ActorCritic(0.11, 0.4)


	x = torch.ones(10, 3, 64, 64)

	y = block(x)
	print(y.shape)

if __name__ == "__main__":
	main()