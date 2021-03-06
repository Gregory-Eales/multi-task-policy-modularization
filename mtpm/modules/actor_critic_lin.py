import torch

class LinActorCritic(torch.nn.Module):

    def __init__(self, actor_lr, epsilon, in_dim, h_dim, out_dim):

        super(LinActorCritic, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim

        self.epsilon = epsilon
        self.define_network()
        #self.optimizer = torch.optim.Adam(params=self.parameters(), lr=actor_lr)
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

        size = self.h_dim

        self.p1 = torch.nn.Linear(self.in_dim, size)
        self.p2 = torch.nn.Linear(size, size)

        self.v1 = torch.nn.Linear(self.in_dim, size)
        self.v2 = torch.nn.Linear(size, size)

        self.pi = torch.nn.Linear(size, self.out_dim)
        self.value = torch.nn.Linear(size, 1)


        self.critic_loss = torch.nn.MSELoss()


    def forward(self, x):


        out = torch.Tensor(x).float().to(self.device)

        p = self.p1(out)
        p = self.tanh(p)

        p = self.p2(p)
        p = self.tanh(p)

        p = self.pi(p)
        pi = self.softmax(p).to(torch.device('cpu:0'))


        v = self.v1(out)
        v = self.tanh(v)
        v = self.v2(v)
        v = self.tanh(v)

        v = self.value(v).to(torch.device('cpu:0'))

        return pi, v
