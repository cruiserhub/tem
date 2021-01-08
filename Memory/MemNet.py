import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import scipy
import pdb
import copy
from scipy.stats import truncnorm
from scipy.special import comb
from Memory.world import Grid, Envs, Agent
from Memory.utils import MLP


# use lstm to retrivial memory and train it? or use hopefield network update?  may find a better readout
class MemoryNet(nn.Module):
    """
    Rey, reused by both Generative and Inference model.
    Generator: pt ~ N(.| u = attractor(gt_, Mt-1), sigma = f(u))
    Inference: px_t = attractor(x_t, Mt-1)
    """
    def __init__(self, n_iteration = 50, kappa = 0.8, lamda = 0.9, yita = 0.1, nx = 26, ng = 17, hop_type = 1, heb_type = 0):
        super().__init__()
        self.n_iteration = n_iteration
        self.kappa = kappa
        self.dim_x = nx
        self.dim_g = ng
        self.hop_type = hop_type
        self.heb_type = heb_type
        self.W_tile = torch.tensor(np.kron(np.ones((1, self.dim_x)), np.eye(self.dim_g)), dtype=torch.float)
        self.W_repeat = torch.tensor(np.kron(np.eye(self.dim_x), np.ones((1, self.dim_g))), dtype=torch.float)
        self.lamda = lamda
        self.yita = yita
        self.M = torch.zeros(1, ng*nx, ng*nx)
        self.r = nn.Parameter(torch.FloatTensor([self.lamda]))
        self.lam_net = MLP(nx * ng, nx * ng, 1)
    """
    Recursive memory retrivial, like a reasnoning process.  The key modification of original hopfield is adding kqv like change  
    """

    def retrive_memory(self, query):
        """
        Generator: pt ~ N(.| u = attractor(gt_, Mt-1), sigma = f(u))
        or Inference: px_t = attractor(xt_, Mt-1)
        :param query: gt_ in Generator or xt_ in Inference
            list, n_stream * (tensor)[batch_size, p_dimension]
        :param M: tensor[batch_size, p_dimension, p_dimension]
        :return: pt(Generator) or px_t(Inference)
            list n_stream * (tensor)[batch_size, p_dimension]
        """
        # dimensionality of h_t is same qs queray
        # h_t = torch.cat(query, dim=1)  # [batch_size, sum(p_dimension_list)]
        h_t = self.f_p(query)  # todo: whether to use leaky relu?
        # todo: check the dimension of the whole function
        # todo: how to keep M upper bound block matrix
        # iteration process to get memory , need to value to be in between -1, 1
        #         print ('query shape', torch.unsqueeze(h_t, 1).shape, 'memory shape', M.shape)
        self.Velocity = []
        for tau in range(self.n_iteration):
            h_t0 = h_t
            # the matrix product between h and M is in dimension  g * x * 1,
            # kappa controls speed to converge to fix point, could change into form of lstm
            if self.hop_type == 0:
                h_t = self.f_p(self.kappa * h_t + torch.squeeze(torch.matmul(torch.unsqueeze(h_t, 1), self.M)))
            elif self.hop_type == 1:
                # this is the attention process like kqv, h is query vector, ht * M is just k * v (memo)
                h_t = self.f_p(self.kappa * h_t + h_t * torch.squeeze(torch.matmul(torch.unsqueeze(h_t, 1), self.M)))
            elif self.hop_type == 2:
                h_t = self.f_p(self.kappa * h_t + h_t * F.softmax(torch.squeeze(torch.matmul(torch.unsqueeze(h_t, 1), self.M))))
            elif self.hop_type == 3:
                h_t = self.f_p(self.kappa * h_t + F.softmax(h_t) * torch.squeeze(torch.matmul(torch.unsqueeze(h_t, 1), self.M)))
            # elif self.hop_type == 3:
            #     h_t = self.f_p(self.r * h_t + h_t * torch.squeeze(torch.matmul(torch.unsqueeze(h_t, 1), self.M)))
            velocity = torch.norm(h_t - h_t0)
            self.Velocity.append(velocity)
        # n_p = np.cumsum(np.concatenate(([0], self.p_dimension_list)))
        # p = [h_t[:, n_p[f]:n_p[f + 1]] for f in range(self.n_stream)]
        return [h_t.reshape(1, -1)]
        # todo: optional? for generative model, Gussian sampling is needed,
        #  while for inference model, it's not

    def f_p(self, input):
        return nn.LeakyReLU()(torch.clamp(input, min=-1, max=1))

    def get_x(self, p_):
        return torch.matmul(self.W_repeat, p_.view(-1))

    def get_g(self, p_):
        return torch.matmul(self.W_tile, p_.view(-1))
    """
    Could use NN to control the information flow of forgetting and writting 
    """
# hebbain learning
    def form_memory(self, p, p_):
        """
        Mt = hebbian(Mt-1, pt)
        Mt = lamda * Mt-1 + yita * (pt - pt_) * (pt + pt_)T
        :param p: pt, list, [batch_size, p_dimension]
        :param p_: pt_, list, [batch_size, p_dimension]
        :param M: Mt-1, [batch_size, p_dimension, p_dimension]
        :return: M: Mt, [batch_size, p_dimension, p_dimension]
        """
        if self.heb_type == 0:
            M_new = torch.squeeze(torch.matmul(torch.unsqueeze(p + p_, 2), torch.unsqueeze(p - p_, 1)))
            self.M = self.lamda * self.M + self.yita * M_new
        elif self.heb_type == 1:
            M_new = torch.squeeze(torch.matmul(torch.unsqueeze(p, 2), torch.unsqueeze(p - p_, 1)))
            self.M = self.lamda * self.M + self.yita * M_new
        elif self.heb_type == 2:
            M_new = torch.squeeze(torch.matmul(torch.unsqueeze(p + p_, 2), torch.unsqueeze(p - p_, 1)))
            self.M = self.lam_net(p) * self.M + self.yita * M_new
        #         print('memory shape', M_new.shape, M.shape)
        # M_new = M_new * self.M_update_mask
        # todo: what' s ovc in parameters.py



class MemoryGame(MemoryNet):
    def __init__(self, type = 'objects', size = 15, replay = 20, n_iteration = 50, kappa = 0.8, lamda = 0.9, yita = 0.1, nx = 26, ng = 17, hop_type = 1, heb_type = 0):
        MemoryNet.__init__(self, n_iteration = n_iteration, kappa = kappa, lamda = lamda, yita = yita, nx = nx, ng = ng, hop_type = hop_type, heb_type=heb_type)
        self.env = Envs(grid_size=size, type = type)
        self.agent = Agent(grid_size=size)
        self.g = []
        self.x = []
        self.size = size
        self.replay = replay

    def generate_walks(self, chunks = 3):
        actions = []
        for i in range(chunks):
            actions = np.random.randint(0, 4, size=13)
        return actions


    def data_collect(self, actions = [], init_pos = (6, 6), chunks = 3, scan = False, sample = False):
        self.x, self.g = [], []
        if scan:
            for p in zip(np.arange(2, self.size +2), np.arange(2, self.size+2)):
                self.agent.pos = p
                pos = torch.eye(self.size + 2)[self.agent.pos[0]] + torch.eye(self.size + 2)[self.agent.pos[1]]
                self.x.append(
                    torch.from_numpy(self.env.sensory[self.agent.pos].ravel()).view(1, -1).type(torch.FloatTensor))
                # print(self.env.objects[agent.pos])
                self.g.append(0.5 * pos.view(1, -1))
        if not scan:
            if sample:
                actions = self.generate_walks(chunks=chunks)
            self.agent.reset(grid_size = self.size, set_agent = init_pos)
            for act in actions:
                # print(act, agent.pos)
                self.agent.act(act)
                pos = torch.eye(self.size + 2)[self.agent.pos[0]] + torch.eye(self.size + 2)[self.agent.pos[1]]
                self.x.append(torch.from_numpy(self.env.sensory[self.agent.pos].ravel()).view(1, -1).type(torch.FloatTensor))
                # print(self.env.objects[agent.pos])
                self.g.append(0.5 * pos.view(1, -1))



    def train(self):
        # take data from environments
        # actions = self.generate_walks(init_pos = (6, 6))
        Loss = 0
        # replay
        for iter in range(self.replay):

            for x, g in zip(self.x, self.g):
                #             print ('item', t)
                # starting point to save process , outter product of x and g
                p = [(torch.t(x) * g).reshape(1, self.dim_x * self.dim_g)]
                #
                p_ = self.retrive_memory(torch.matmul(g, self.W_tile))
                #             print('place cell of retrivial', len(p_), p_[0].shape)
                #  memory formation is just to save difference between seen data and unseen
                self.form_memory(p[0], p_[0])
                loss = torch.sum(torch.abs(p[0] - p_[0]))
                # loss.backward(retain_graph=True)

                Loss += loss
        # Optim.step()
        # Optim.zero_grad()
        return Loss

    def train_slow(self):
        # take data from environments
        # actions = self.generate_walks(init_pos = (6, 6))
        Loss = 0
        # replay
        Optim = torch.optim.Adam(self.lam_net.parameters(), lr=1e-3)
        mse = torch.nn.MSELoss()
        for iter in range(self.replay):

            for i, (x, g) in enumerate(zip(self.x, self.g)):
                #             print ('item', t)
                # starting point to save process , outter product of x and g
                with torch.autograd.set_detect_anomaly(True):
                    p = [(torch.t(x) * g).reshape(1, self.dim_x * self.dim_g)]
                    #
                    p_ = self.retrive_memory(torch.matmul(g, self.W_tile))
                    #             print('place cell of retrivial', len(p_), p_[0].shape)
                    #  memory formation is just to save difference between seen data and unseen
                    print ('p', p_[0].shape)
                    self.form_memory(p[0], p_[0])
                    # if i>= 1:
                    loss = mse(p_[0], p[0])
                    loss.backward()
                    Loss += loss
                    Optim.step()
                    Optim.zero_grad()
        return Loss


    def test(self, show = False):
        Acc = 0
        for x, g in zip(self.x, self.g):
            p_ = self.retrive_memory(torch.matmul(g, self.W_tile))
            wordcode = torch.matmul(p_[0], torch.t(self.W_repeat)).data.numpy().ravel()
            predict = np.argmax(self.env.decode_nhot(wordcode))
            label = np.argmax(self.env.decode_nhot(x.data.numpy().ravel()))
            Acc += (predict == label)
            print (predict, label)
            if show:
                print('predict', self.env.objects_type[predict])
                print('label', self.env.objects_type[label])
                print('x', torch.topk(x, k=4)[1], 'readout x',
                      torch.topk(torch.matmul(p_[0], torch.t(self.W_repeat)), k=4)[1], "\\n")
                print('x', torch.round(torch.topk(x, k=4)[0]), 'readout x', torch.round(torch.topk(torch.matmul(p_[0], torch.t(self.W_repeat)), k=4)[0]), "\\n")
        Acc = Acc / len(self.x)
        print('accuracy', Acc)
        return Acc

"""
it is easy to forget, once run two traces, the last one cannot be recalled (yita = 0.5, 0.1). problem should due to storage
(p- p_) should return 0 if the two are similar, it may be due to poor memory capacity, increasing the length of each random works
seems improves the performance.

Analyse can be done for 1, evns type(is it pass hopfield limit?) 2, evns correlation, is it spurious correlation? 3, stabilization with 
time or places visited or neither?  Reduce replay of each episode improves , reduce map size improves
"""


# initial replay 5, kappa 0.9
if __name__ == '__main__':
    writer = SummaryWriter('/home/tie/tem/experiment_5')
    size =7
    game = MemoryGame(size = size, n_iteration=5, replay = 1, kappa = 0.5, lamda = 0.99, yita = 0.5, hop_type=1, heb_type = 0, nx = 26, ng = size+2)
    buffer_x, buffer_g = [], []
    iters = 20
    for i in range(100):
        M = game.M
        game.yita = 0.5
        game.data_collect(scan=False, sample=True, init_pos=(np.random.randint(3, 3 + size//2), np.random.randint(3, 3 + size//2)), chunks=5)
        loss = game.train()
        writer.add_scalar('training loss',
                          loss ,i)
        # print (i, torch.norm(game.M - M), 'train loss', loss)
        buffer_x.append(game.x)
        buffer_g.append(game.g)
        sample = True
        if i%10 == 0:
            if sample:
                game.x, game.g = [], []
                game.data_collect(scan = False, sample = True, chunks = 3)
                Acc = game.test()
                writer.add_scalar('Accuracy',
                                  Acc, i)
                writer.close()
            else:
                game.x, game.g = [], []
                # ind = np.random.randint(len(buffer_g))
                inds = [-1, 0]
                for ind in inds:
                    print('ind', ind)
                    game.x = buffer_x[ind]
                game.g = buffer_g[ind]
                game.test()






