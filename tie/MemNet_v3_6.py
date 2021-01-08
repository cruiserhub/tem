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
from world import Grid, Envs, Agent
from utils import MLP
from torch.utils.data import Dataset, DataLoader
import itertools


# use lstm to retrivial memory and train it? or use hopefield network update?  may find a better readout
class MemoryNet(nn.Module):
    """
    Rey, reused by both Generative and Inference model.
    Generator: pt ~ N(.| u = attractor(gt_, Mt-1), sigma = f(u))
    Inference: px_t = attractor(x_t, Mt-1)
    """
    def __init__(self, n_iteration = 50, kappa = 0.8, lamda = 0.9, yita = 0.1, nx_in = 26, ng_in = 17, nx_out = 20, ng_out = 10, num_class = 7, hop_type = 1, heb_type = 0):
        super().__init__()
        self.n_iteration = n_iteration
        self.kappa = kappa
        self.dim_x = nx_out
        self.dim_g = ng_out
        self.hop_type = hop_type
        self.heb_type = heb_type
        self.W_tile = torch.tensor(np.kron(np.ones((1, self.dim_x)), np.eye(self.dim_g)), dtype=torch.float)
        self.W_repeat = torch.tensor(np.kron(np.eye(self.dim_x), np.ones((1, self.dim_g))), dtype=torch.float)
        self.lamda = lamda
        self.yita = yita
        self.register_buffer('M', torch.zeros(1, self.dim_g*self.dim_x, self.dim_g*self.dim_x))
        self.r = nn.Parameter(torch.FloatTensor([self.lamda]))
        # self.lam_net = MLP(nx * ng_out, nx * ng_out, 1)
        self.x_transform = True
        self.g_transform = True
        self.transform_g = MLP(in_dim=ng_in, hidden_dim=2* ng_in,  out_dim=ng_out)
        self.transform_sensory = MLP(in_dim=nx_in, hidden_dim=2 * nx_in, out_dim=nx_out)
        # self.predict_sensory = nn.Sequential(nn.Linear(nx_out, num_class))
        self.predict_sensory = MLP(in_dim=nx_out, out_dim=num_class, bias=[True], num_layer=1)
        self.relu = nn.LeakyReLU(inplace=False)
        self.tanh = nn.Tanh()
        self.x_buffer = []
        self.g_buffer = []
        self.label_buffer = []

    """
    Recursive memory retrivial, like a reasnoning process.  The key modification of original hopfield is adding kqv like change  
    """
   # checked with no problem, key variables, p_, M
    def retrive_memory(self, query, M):
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

        for tau in range(self.n_iteration):
            # the matrix product between h and M is in dimension  g * x * 1,
            # kappa controls speed to converge to fix point, could change into form of lstm
            if self.hop_type == 0:
                h_t = self.f_p(self.kappa * h_t + torch.squeeze(torch.matmul(torch.unsqueeze(h_t, 1), M)))
            elif self.hop_type == 1:
                # this is the attention process like kqv, h is query vector, ht * M is just k * v (memo)
                h_t = self.f_p(self.kappa * h_t + h_t * torch.squeeze(torch.matmul(torch.unsqueeze(h_t, 1), M)))
            elif self.hop_type == 2:
                h_t = self.f_p(self.kappa * h_t + h_t * F.softmax(torch.squeeze(torch.matmul(torch.unsqueeze(h_t, 1), M))))
            # elif self.hop_type == 3:
            # print (self.M.shape)
            # h_t = torch.squeeze(torch.matmul(torch.unsqueeze(h_t, 0), self.M))
        # h_t = h_t.matmul(M)
            # h_t = query
            # elif self.hop_type == 3:
            #     h_t = self.f_p(self.r * h_t + h_t * torch.squeeze(torch.matmul(torch.unsqueeze(h_t, 1), self.M)))
        # n_p = np.cumsum(np.concatenate(([0], self.p_dimension_list)))
        # p = [h_t[:, n_p[f]:n_p[f + 1]] for f in range(self.n_stream)]
        return h_t.reshape(1, 1, -1)
        # todo: optional? for generative model, Gussian sampling is needed,
        #  while for inference model, it's not

    def f_p(self, input):
        x = self.relu(input)
        # print (x)
        x = torch.clamp(x, min=-1, max=1)

        return x

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
        M_new = torch.squeeze(torch.matmul(torch.unsqueeze(p + p_, 2), torch.unsqueeze(p - p_, 1)))
        if self.heb_type == 0:
            self.M = self.lamda * self.M + self.yita * M_new
        elif self.heb_type == 1:
            self.M = self.lam_net(p) * self.M + self.yita * M_new

        #         print('memory shape', M_new.shape, M.shape)
        # M_new = M_new * self.M_update_mask
        # todo: what' s ovc in parameters.py

    def clear_memory(self):
        self.M = torch.zeros(1, self.dim_g*self.dim_x, self.dim_g*self.dim_x)
        self.x_buffer = []
        self.g_buffer = []
        self.label_buffer = []

    def forward(self, x, g, M, heb = True):
        if self.x_transform:
            x_ = self.transform_sensory(x)
        else:
            x_ = x
        if self.g_transform:
            g_ = self.transform_g(g)
        else:
            g_ = g
        # print (x_.shape, g_.shape)
        p = [torch.matmul(torch.t(x_), g_) .reshape(1, self.dim_x * self.dim_g)]
        #
        p_ = self.retrive_memory(torch.matmul(g_, self.W_tile), M)

        #             print('place cell of retrivial', len(p_), p_[0].shape)
        #  memory formation is just to save difference between seen data and unseen
        if heb:
            self.form_memory(p[0], p_[0])
        x_out = torch.matmul(self.W_repeat, p_.squeeze())
        x_inf = self.predict_sensory(x_out)

        return x_inf, g_, x_, p, p_



class MemoryGame(MemoryNet):
    def __init__(self, type = 'objects', size = 15, n_iteration = 50, kappa = 0.8, lamda = 0.9, yita = 0.1, nx_in = 26, nx_out = 10, ng_in = 17, ng_out = 10, num_class = 7, hop_type = 1, heb_type = 0):
        MemoryNet.__init__(self, n_iteration = n_iteration, kappa = kappa, lamda = lamda, yita = yita, nx_in = nx_in, nx_out = nx_out, ng_in = ng_in,  ng_out = ng_out, num_class = num_class, hop_type = hop_type, heb_type=heb_type)
        self.env = Envs(grid_size=size, type = type)
        self.agent = Agent(grid_size=size)
        self.g = []
        self.x = []
        self.labels = []
        self.size = size
        self.type = type
        self.batch_size = 1

    def reset_env(self):
        self.env = Envs(grid_size=self.size, type=self.type)

    def generate_walks(self, chunks=3):

        actions = [[np.random.randint(0, 4)] * 5 for i in range(chunks)]
        actions = list(itertools.chain.from_iterable(actions))
        return actions


    def data_collect(self, chunks = 3):
        Pos = zip([np.random.randint(2, self.size+2) for i in range(5)], [np.random.randint(2, self.size+2) for i in range(5)])
        actions = self.generate_walks(chunks=chunks)
        # print (Pos)
        for init_pos in Pos:
            # print (init_pos)
            self.agent.reset(grid_size = self.size, set_agent = init_pos)
            for act in actions:
                # print(act, agent.pos)
                self.agent.act(act)
                pos = torch.eye(self.size + 2)[self.agent.pos[0]] + torch.eye(self.size + 2)[self.agent.pos[1]]
                self.x_buffer.append(
                    torch.from_numpy(self.env.sensory[self.agent.pos].ravel()).view(1, -1).type(torch.FloatTensor))
                # print(self.env.objects[agent.pos])
                self.g_buffer.append(0.5 * pos.view(1, -1))
                self.label_buffer.append(torch.LongTensor(self.env.objects[self.agent.pos]))
                # print (self.agent.pos, act, self.env.objects[self.agent.pos])

    """
     p_ and M are used in a recursive manner, with computation graph retained , all variables in loop should be kept same until the graph released, or stop gradient there 
    """

    # attention to loss, should
    def train_g(self, xs, gs, labels, heb = True, shuffle = False):
        # take data from environments
        # actions = self.generate_walks(init_pos = (6, 6))
        # if not shuffle:
        #     self.x = torch.stack(self.x)
        #     self.g = torch.stack(self.g)
        #     self.labels = torch.stack(self.labels)
        # else:
        #     train_orders = np.arange(len(self.x))
        #     np.random.shuffle((train_orders))
        #     self.x = self.x[train_orders]
        #     self.g = self.g[train_orders]
        #     self.labels = self.labels[train_orders]
            # print (train_orders)
        Loss, Loss_heb = [], []
        # replay
        Optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        mse = torch.nn.MSELoss()
        # it contains softmax so no activation function before
        entropy = torch.nn.CrossEntropyLoss()
        # M = np.zeros((1, self.dim_x*self.dim_g, self.dim_x*self.dim_g))
        # p_ = np.zeros((1, 1, self.dim_x * self.dim_g))
        # M = torch.from_numpy(M).type('torch.FloatTensor')
        # p_ = torch.from_numpy(p_).type('torch.FloatTensor')

            # p_ = torch.zeros(1, 1, 234, requires_grad=False)
        # self.M.require_grad = True
        X_inf, Label = [], []
        # self.M.detach_()
        self.M = torch.autograd.Variable(self.M)
        # heb is a serial process(formation and retrivial), other than that all can be parallized
        for i, (x, g, label) in enumerate(zip(xs, gs, labels)):
            #             print ('item', t)
            # leaf variable M, x, g, so no intermidate variables will be used
            x_inf, g_, x_ , p, p_= self.forward(x, g, self.M, heb=heb)
            # for item in x_inf, g_, x_ , p, p_, x, g:
            #     if type(item) is list:
            #         print(len(item), item[0].shape)
            #     else:
            #         print(item.shape)
            # print('---------------')
            # print(x, g)
            # try to get softmax of num of word if want entropy
            # print ('loss', x_inf.view(self.batch_size, -1), torch.FloatTensor(label))
            # l1 loss
            # loss_reg = 0
            # for par in self.parameters():
            #     loss_reg += torch.abs(par).sum()
            # loss_heb = mse(p_[0], p[0])
            X_inf.append(x_inf)
            Label.append(label)
            # print (label)
            # loss = entropy(x_inf.view(self.batch_size, -1), torch.FloatTensor(label).view(self.batch_size)) #+ 1e-2 * loss_reg
            # Loss_heb.append(loss_heb)
            # Loss.append(loss)
            # p_ = p_.detach()
            # p_.requires_grad = False
            # M = M.data.numpy()
            # p_ = p_.data.numpy()
            # print ('grad', self.predict_sensory[0].weight.grad)
            # weight = self.predict_sensory[0].weight.data.clone()
        # Loss = torch.sum(torch.stack(Loss))
        # Loss_heb = torch.sum(torch.stack(Loss_heb))
        Optim.zero_grad()
        X_inf = torch.stack(X_inf)
        Label = torch.stack(Label)
        # back prop on inference of x
        Loss = entropy(X_inf.view(len(X_inf), -1), Label.view(len(Label)))
        Loss.backward()
        # Loss_heb.backward(retain_graph=True)
        Optim.step()
        # print('change', weight - self.predict_sensory[0].weight.data)

        self.M = self.M.detach()
        # p_ = p_.detach()
        return Loss, Loss_heb


    def test(self, show = False):
        Acc = 0
        for x, g in zip(self.x, self.g):
            p_ = self.retrive_memory(torch.matmul(g, self.W_tile))
            wordcode = torch.matmul(p_[0], torch.t(self.W_repeat)).data.numpy().ravel()
            predict = np.argmax(self.env.decode_nhot(wordcode))
            label = np.argmax(self.env.decode_nhot(x.data.numpy().ravel()))
            Acc += (predict == label)
            if show:
                pass
                # print('predict', self.env.objects_type[predict])
                # print('label', self.env.objects_type[label])
                # print('x', torch.topk(x, k=4)[1], 'readout x',
                #      torch.topk(torch.matmul(p_[0], torch.t(self.W_repeat)), k=4)[1], "\\n")
                # print('x', torch.round(torch.topk(x, k=4)[0]), 'readout x', torch.round(torch.topk(torch.matmul(p_[0], torch.t(self.W_repeat)), k=4)[0]), "\\n")
        Acc = Acc / len(self.x)
        # print('accuracy', Acc)
        return Acc

    def test_g(self):
        Acc = 0
        indice = torch.tensor(np.random.randint(len(self.x_buffer), size = 20))
        # print ('len', len(game.x_buffer), indice)
        xs = torch.stack(self.x_buffer)[indice]
        gs = torch.stack(self.g_buffer)[indice]
        labels = torch.stack(game.label_buffer)[indice]
        self.M.detach_()
        for x, g, label in zip(xs, gs, labels):
            x_inf, g_, x_, _, _ = self.forward(x, g, self.M, heb = False)
            # print ('g', self.transform_g(g))
            # print (x_)
            predict = torch.argmax(x_inf)
            # print (predict, label)
            Acc += (predict.data.numpy() == label.data.numpy())
        Acc = Acc / len(xs)
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
    # writer = SummaryWriter('/home/tie/tem/experiment_6')
    size =10
    replay = 10
    game = MemoryGame(size = size, n_iteration=2, kappa = 0.5, lamda = 0.8, yita = 0.5, hop_type=1, heb_type = 0, nx_in = 26, ng_in = size+2, nx_out = 15, ng_out = 15, num_class=8)
    for par in game.parameters():
        # print (par.shape)
        pass
    # buffer_x, buffer_g = [], []
    Loss, Loss_heb = 0, 0
    for i in range(int(1e4)):
        if i%10 == 0 and i>0:
            game.M.detach()
            game.clear_memory()
            game.reset_env()
        game.data_collect()
        # train_data = zip(game.x, game.g, game.labels)
        # train_loader = DataLoader(dataset=train_data, shuffle=True)
        indice = torch.tensor(np.random.randint(len(game.x_buffer), size = 20))
        # print ('len', len(game.x_buffer), indice)
        xs = torch.stack(game.x_buffer)[indice]
        gs = torch.stack(game.g_buffer)[indice]
        labels = torch.stack(game.label_buffer)[indice]
        loss, _ = game.train_g(xs, gs, labels)
        Loss += loss
        # for k in range(replay):
        #     loss, loss_heb = game.train_g(shuffle=True, heb=False)
        #     Loss += loss
        #     Loss_heb += loss_heb
        # # writer.add_scalar('training loss', loss ,i%10)
        # print (i, torch.norm(game.M - M), 'train loss', loss)
        # paras = torch.cat([x.view(-1) for x in game.transform_g.parameters()])
        # print (torch.norm(paras))
        # buffer_x.append(game.x)
        # buffer_g.append(game.g)
        if i%10 == 0:
            # print ('loss', Loss)
            # torch.save(game, '/mnt/data/tie/tem/tem.pth')
            Loss, Loss_heb = 0, 0
        Acc = 0
        if (i+1)%10 == 0 and i>0:
            sample = True
            if sample:
                game.x, game.g = [], []
                game.data_collect()
                # print ('epoch', i%10)
                acc = game.test_g()
                # writer.add_scalar('Accuracy', acc, i)
                Acc += acc
                # writer.close()
        if (i+1)%100 == 0:
            # print ('avg acc', Acc/10)
            Acc = 0
            for par in game.predict_sensory.parameters():
                # print(par)
                pass
            # else:
            #     game.x, game.g = [], []
            #     # ind = np.random.randint(len(buffer_g))
            #     inds = [-1, 0]
            #     for ind in inds:
            #         print('ind', ind)
            #         game.x = buffer_x[ind]
            #         game.g = buffer_g[ind]
            #         game.test_g()






