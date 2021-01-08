#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is for testing the basic memory of TEM

@author: tlk
"""
import torch
import torch.nn as nn
import random
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
from model2 import MLP, CompressSensory, Sensory2Hippo, \
    Entor2Hipp, SensoryPrediction, InferHipp
from MemNet_v3_6 import MemoryGame


class Attention(nn.Module):
    def __init__(self, n_head, d_k0, d_v0, d_k, d_v):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        # dv0 = n_head * d_v
        self.wq = nn.Linear(d_k0, n_head * d_k, bias=False)
        self.wk = nn.Linear(d_k0, n_head * d_k, bias=False)
        self.wv = nn.Linear(d_v0, n_head * d_v, bias=False)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, query, key, value):
        """
        :param query: [batch_size, seq_len1, d_model]  [batch_size, d_model]
        :param key: [batch_size, seq_len2, d_model]  [batch_size, N, d_model]
        :param value: [batch_size, N, d_model]
        :return: [batch_size, d_v0]
        """
        query = query.unsqueeze(1)
        q = self.wq(query).view(*query.shape[0:2], self.n_head, self.d_k).permute(0, 2, 1, 3) # [b, h, 1, dk]
        k = self.wk(key).view(*key.shape[0:2], self.n_head, self.d_k).permute(0, 2, 3, 1) # [b, h, dk, s2]
        v = self.wv(value).view(*key.shape[0:2], self.n_head, self.d_v).permute(0, 2, 1, 3) # [b, h, s2, dv]
        attn = self.softmax((torch.matmul(q, k) / math.sqrt(self.d_k)))
        return attn.matmul(v).view(-1, self.n_head * self.d_v)  # [b, 8, 1, s2] * [b, 8, s2, 64] => [b, h, 1, dv]


def generating_pairs(x_dim, g_dim, n_pairs):
    # return [[x1, g1], [x2, g2], ...]
    x = torch.stack([torch.eye(x_dim)[random.randint(0, x_dim - 1)]
                      for _ in range(n_pairs)], dim=0)
    g = 0.5 * torch.clamp(torch.randn(n_pairs, g_dim), -2, 2)
    zero_mask = torch.randn(n_pairs, g_dim) + 0.1
    zero_mask = torch.where(zero_mask < 0,
                            torch.full_like(zero_mask, 0), torch.full_like(zero_mask, 1))
    # zero_mask = 1
    g = g * zero_mask
    return x, g

def generating_pairs_tie(n_pairs, seq_len):
    """
    MemoryGame is only used to generate pairs
    return seq_len of n x-g pairs from the same envs
    if the func is called twice, the envs will change
    :param n_pairs: batch_size
    :return: xs s*[b, 26], gs s*[b, 12], labels s*[b,]
    """
    size = 10
    game = MemoryGame(size = size, n_iteration=2, kappa = 0.5,
                      lamda = 0.8, yita = 0.5, hop_type=1, heb_type = 0,
                      nx_in = 26, ng_in = size+2, nx_out = 15, ng_out = 15, num_class=8)
    game.reset_env()
    x_samples = []
    gs_samples = []
    labels = []
    for _ in range(seq_len):
        game.clear_memory()
        game.data_collect()
        indice = torch.tensor(np.random.randint(len(game.x_buffer), size=n_pairs))
        xs = (torch.stack(game.x_buffer)[indice]).squeeze().detach()
        x_samples.append(xs)
        gs = (torch.stack(game.g_buffer)[indice]).squeeze().detach()
        # ------------------
        # g = 0.5 * torch.clamp(torch.randn(n_pairs, 12), -2, 2)
        # zero_mask = torch.randn(n_pairs, 12) - 1
        # zero_mask = torch.where(zero_mask < 0,
        #                         torch.full_like(zero_mask, 0), torch.full_like(zero_mask, 1))
        # g = g * zero_mask
        # -----------------
        gs_samples.append(gs)
        labels.append((torch.stack(game.label_buffer)[indice]).squeeze().detach())
    return x_samples, gs_samples, labels

class InferEntor(nn.Module):
    """
    Infer Entorhinal
    gt = clamp(mlp(px))
    """
    def __init__(self, g_downsampled_dimension, g_dimension, xf_dimension):
        super().__init__()
        # W_tile [g_downsampled_dimension, p_dimension]
        self.W_repeat = nn.Parameter(torch.tensor(np.kron(np.eye(g_downsampled_dimension),
                                             np.ones((1, xf_dimension))), dtype=torch.float),
                                     requires_grad=False)
        self.mlp_g_mean_correction = MLP(
            in_dim=g_downsampled_dimension, out_dim=g_dimension,
            activation=(nn.functional.elu, None), hidden_dim=2 * g_dimension,
            bias=(True, True)
        )


    def forward(self, px):
        mlp_input = torch.matmul(px, torch.t(self.W_repeat))
        g_mean_correction = self.mlp_g_mean_correction(mlp_input)
        g_mean_correction = torch.clamp(g_mean_correction, -1, 1)
        return g_mean_correction

class TEM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x_dim = 8  # 8 is class num, 45
        self.xc_dim = 26  # 26 is x's dim, 10
        self.g_dim = 12  # 20
        self.p_dim = self.xc_dim * self.g_dim
        self.n_iteration = 5

        self.attention = Attention(n_head=1, d_k0=self.g_dim, d_v0=self.xc_dim,
                                   d_k=12, d_v=26)  # todo: d_k, d_v tunable
        self.compress_sensory = CompressSensory(self.x_dim, self.xc_dim)
        self.mlp_x2x = MLP(
            in_dim=self.xc_dim, out_dim=self.xc_dim,
            activation=(nn.functional.elu, None), hidden_dim=2 * self.xc_dim,
            bias=(True, True)
        )
        self.sensory2hippo = Sensory2Hippo(self.g_dim, self.xc_dim)

        self.infer_entor = InferEntor(self.g_dim, self.g_dim, self.xc_dim)

        self.mlp_g2g = MLP(
            in_dim=self.g_dim, out_dim=self.g_dim,
            activation=(nn.functional.elu, None), hidden_dim=2 * self.g_dim,
            bias=(True, True)
        )
        self.entor2hipp = Entor2Hipp(self.g_dim, self.xc_dim, self.g_dim)
        self.sensory_prediction = SensoryPrediction(self.g_dim, self.xc_dim,
                                                    self.x_dim, hidden_dim=None)
        self.infer_hipp = InferHipp()
        self.mlp_predict_x = MLP(
            in_dim=self.xc_dim, out_dim=self.x_dim,
            activation=(nn.functional.elu, None), hidden_dim=2 * self.xc_dim,
            bias=(True, True)
        )

        self.Mx = None
        self.Mg = None

    def forward(self, x, g):
        # x[b, x_dim], g[b, g_dim]
        # ------- infer g from x ---------
        xc = x  # self.compress_sensory(x)  # one_hot to two_hot
        # xc = self.mlp_x2x(xc)      # mlp  # todo
        x_ = xc  # self.sensory2hippo(xc)    # wp, x_ = W_tile * wp * fn(xf)
        # px = self.attention()
        # g_inf = self.infer_entor(px[0])  # gt = clamp(mlp(px))

        # ------- infer x from g ---------
        # g = torch.tanh(self.mlp_g2g(g))   # todo: other normalization methods
        g_ = g  # self.entor2hipp(g)   # gt_ = W_repeat * f_down(g)

        if self.Mg is None or self.Mx is None:
            self.Mg = g_.unsqueeze(1).clone().detach()
            self.Mx = x_.unsqueeze(1).clone().detach()
            return None, None, g_, x_
        retrived_x = self.attention(query=g_, key=self.Mg, value=self.Mx)
        x_inf = self.mlp_predict_x(retrived_x)  # mlp
        self.Mg = torch.cat((self.Mg, g_.unsqueeze(1)), dim=1)  # [b, N, xc] [b, 1, xc]
        self.Mx = torch.cat((self.Mx, x_.unsqueeze(1)), dim=1)

        return None, x_inf, g_, x_

if __name__ == '__main__':
    lr = 1e-3 # 9.4e-4
    epoches = 5000
    batch_size = 100
    view_times = 2  # todo
    seq_len_train = 50
    seq_len_test = 1
    # train_x_samples, train_g_samples = zip(*[generating_pairs(x_dim=45, g_dim=20, n_pairs=batch_size) for _ in range(seq_len_train)])
    # test_x_samples, test_g_samples = zip(*[generating_pairs(x_dim=45, g_dim=20, n_pairs=batch_size) for _ in range(seq_len_test)])
    train_x_samples, train_g_samples, train_labels = generating_pairs_tie(batch_size, seq_len_train)
    # todo: test was wrong, because generating_pairs are called twice
    test_x_samples, test_g_samples, test_labels = train_x_samples[0:seq_len_test], train_g_samples[0:seq_len_test], train_labels[0:seq_len_test]
    tem = TEM()
    # with open('tem.pth', 'rb') as f:
    #     tem = torch.load(f)
    tem.Mg = tem.Mx = None
    optim = adam = torch.optim.Adam(tem.parameters(), lr = lr)

    losses_x = []
    losses_g = []
    losses_reg_x = []
    losses_reg_g = []
    train_accuracys = []
    eval_accuracys = []
    for i in range(epoches):
        # ---------train---------------
        train_orders = [p for p in range(seq_len_train)]
        np.random.shuffle((train_orders))
        train_acc = 0
        if tem.Mg is not None and tem.Mx is not None:
            tem.Mg = tem.Mg.detach()
            tem.Mx = tem.Mx.detach()
        epoch_loss_x = []
        epoch_loss_g = []
        epoch_loss_reg_x = []
        epoch_loss_reg_g = []
        if i % view_times == 0:
            tem.Mg = tem.Mx = None
            # todo: update train samples every view_tiems steps
            train_x_samples, train_g_samples, train_labels = generating_pairs_tie(batch_size, seq_len_train)
            test_x_samples, test_g_samples, test_labels = train_x_samples[0:seq_len_test], train_g_samples[0:seq_len_test], train_labels[0:seq_len_test]

        for j in train_orders:
            train_x = train_x_samples[j]
            train_g = train_g_samples[j]
            train_x_labels = train_labels[j]
            g_inf, x_inf, g_, x_ = tem(train_x, train_g)
            if i % view_times != 0:
                # x_labels = torch.argmax(train_x, dim=-1)
                loss_x = torch.nn.CrossEntropyLoss()(x_inf, train_x_labels)  # todo
                loss_g = torch.zeros(1) # torch.nn.MSELoss()(g_inf, train_g)  # todo
                # todo: tentatively add classification loss---------
                # loss_reg_x = torch.nn.CrossEntropyLoss()(tem.mlp_predict_x(train_x), train_x_labels)
                loss_reg_x = torch.zeros(1) # torch.sum(torch.abs(x_))
                loss_reg_g = torch.zeros(1) # torch.sum(torch.abs(g_))
                epoch_loss_x.append(loss_x)
                epoch_loss_g.append(loss_g)
                epoch_loss_reg_x.append(loss_reg_x)
                epoch_loss_reg_g.append(loss_reg_g)

                train_acc += (np.argmax(x_inf.detach().numpy(), axis=-1)
                    == train_x_labels.numpy()).astype(np.int16).mean()
        if i % view_times != 0:
            optim.zero_grad()
            epoch_loss_x = torch.sum(torch.stack(epoch_loss_x))
            epoch_loss_g = torch.sum(torch.stack(epoch_loss_g))
            epoch_loss_reg_x = torch.sum(torch.stack(epoch_loss_reg_x))
            epoch_loss_reg_g = torch.sum(torch.stack(epoch_loss_reg_g))

            epoch_loss = epoch_loss_x # todo: tuning
            epoch_loss.backward()
            optim.step()
            losses_x.append(epoch_loss_x.item())
            losses_g.append(epoch_loss_g.item())
            losses_reg_x.append(epoch_loss_reg_x.item())
            losses_reg_g.append(epoch_loss_reg_g.item())
            train_accuracys.append(train_acc / seq_len_train)
        # ---------------------eval-----------------
        eval_acc = 0
        for j in range(seq_len_test):
            _, x_inf, _, _ = tem(test_x_samples[j],
                           test_g_samples[j])
            eval_acc += (np.argmax(x_inf.detach().numpy(), axis=-1)
                         == test_labels[j].numpy()).astype(np.int16).mean()
        if i % view_times != 0:
            eval_accuracys.append(eval_acc / seq_len_test)

        # --------------------plot-----------------
        if i % 10 == 1:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()
            ax2.plot(train_accuracys, color='r')
            ax2.plot(eval_accuracys, color='y')
            for loss_recording in [losses_x, losses_g, losses_reg_x, losses_reg_g]:
                raw = np.array(loss_recording)
                raw = (raw - np.min(raw)) / (np.max(raw) - np.min(raw) + 1e-9)
                ax1.plot(raw)
            ax1.legend(['Loss_x', 'Loss_g', 'Loss_reg_x', 'Loss_reg_g'], loc='upper left')
            ax2.legend(['train', 'eval'], loc='upper right')
            plt.savefig('trainning_curve.png')
            plt.show()
            print('saving------------------')
            torch.save(tem, 'tem.pth')
            with open('tem.pkl', 'wb') as f:
                log = {
                    'loss_x': losses_x,
                    'loss_g': losses_g,
                    'loss_reg_x': losses_reg_x,
                    'loss_reg_g': losses_reg_g,
                    'xc_mlp': False,
                }
            print('ok')


