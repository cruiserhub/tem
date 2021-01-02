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
import matplotlib.pyplot as plt
from model2 import MLP, CompressSensory, Sensory2Hippo, RetriveMemory, \
    Entor2Hipp, SensoryPrediction, FormMemory, InferHipp
import pickle

def generating_pairs(x_dim, g_dim, n_pairs):
    # return [[x1, g1], [x2, g2], ...]
    x = torch.stack([torch.eye(x_dim)[random.randint(0, x_dim - 1)]
                      for _ in range(n_pairs)], dim=0)
    g = 0.5 * torch.clamp(torch.randn(n_pairs, g_dim), -2, 2)
    zero_mask = torch.randn(n_pairs, g_dim) + 1
    zero_mask = torch.where(zero_mask < 0,
                            torch.full_like(zero_mask, 0), torch.full_like(zero_mask, 1))
    # zero_mask = 1
    g = g * zero_mask
    return x, g

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
        self.x_dim = 45
        self.xc_dim = 10
        self.g_dim = 20
        self.p_dim = self.xc_dim * self.g_dim
        self.n_iteration = 5

        self.compress_sensory = CompressSensory(self.x_dim, self.xc_dim)
        self.mlp_x2x = MLP(
            in_dim=self.xc_dim, out_dim=self.xc_dim,
            activation=(nn.functional.elu, None), hidden_dim=2 * self.xc_dim,
            bias=(True, True)
        )
        self.sensory2hippo = Sensory2Hippo(self.g_dim, self.xc_dim)
        self.retrieve_memory = RetriveMemory(n_stream=1, n_iteration=self.n_iteration,
                                             p_dimension_list=[self.p_dim],
                                             kappa=0.8, retrive_it_mask=False)
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
        self.form_memory = FormMemory(n_stream=1, p_dimension_list=[self.p_dim],
                                      lamda=0.9999, yita=0.5)
        self.register_buffer('M', torch.zeros(1, self.p_dim, self.p_dim))

    def forward(self, x, g):
        # x[b, x_dim], g[b, g_dim]
        # ------- infer g from x ---------
        xc = self.compress_sensory(x)  # one_hot to two_hot
        # xc = self.mlp_x2x(xc)      # mlp  # todo
        x_ = self.sensory2hippo(xc)    # wp, x_ = W_tile * wp * fn(xf)
        px = self.retrieve_memory([x_], self.M)
        g_inf = self.infer_entor(px[0])  # gt = clamp(mlp(px))

        # ------- infer x from g ---------
        g = torch.tanh(self.mlp_g2g(g))   # todo: other normalization methods
        g_ = self.entor2hipp(g)   # gt_ = W_repeat * f_down(g)
        p = self.retrieve_memory([g_], self.M)
        x_inf = self.sensory_prediction(p[0])  #  x = mlp(wx * W_tile * pt + bx)

        p_inf = self.infer_hipp(g_, x_)
        self.M = self.form_memory([p_inf], p, self.M)
        return g_inf, x_inf, g_, x_

if __name__ == '__main__':
    lr = 1e-3 # 9.4e-4
    epoches = 1000
    batch_size = 100
    view_times = 5
    seq_len_train = 50
    seq_len_test = 2
    train_x_samples, train_g_samples = zip(*[generating_pairs(x_dim=45, g_dim=20, n_pairs=batch_size) for _ in range(seq_len_train)])
    test_x_samples, test_g_samples = zip(*[generating_pairs(x_dim=45, g_dim=20, n_pairs=batch_size) for _ in range(seq_len_test)])
    tem = TEM()
    # with open('tem.pth', 'rb') as f:
    #     tem = torch.load(f)
    tem.M = torch.zeros_like(tem.M)
    optim = adam = torch.optim.Adam(tem.parameters(), lr = lr)

    # todo: train like continuous learning
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
        tem.M = tem.M.detach()
        epoch_loss_x = []
        epoch_loss_g = []
        epoch_loss_reg_x = []
        epoch_loss_reg_g = []
        if i % view_times == 0:
            tem.M = torch.zeros_like(tem.M)

        for j in train_orders:
            train_x = train_x_samples[j]
            train_g = train_g_samples[j]
            g_inf, x_inf, g_, x_ = tem(train_x, train_g)
            if i % view_times != 0:
                x_labels = torch.argmax(train_x, dim=-1)
                loss_x = torch.nn.CrossEntropyLoss()(x_inf, x_labels)  # todo
                loss_g = torch.nn.MSELoss()(g_inf, train_g)  # todo
                loss_reg_x = torch.sum(torch.abs(x_))
                loss_reg_g = torch.sum(torch.abs(g_))
                epoch_loss_x.append(loss_x)
                epoch_loss_g.append(loss_g)
                epoch_loss_reg_x.append(loss_reg_x)
                epoch_loss_reg_g.append(loss_reg_g)

                train_acc += (np.argmax(x_inf.detach().numpy(), axis=-1)
                    == np.argmax(train_x.detach().numpy(), axis=-1)).mean()
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
                         == np.argmax(test_x_samples[j].unsqueeze(0).detach().numpy(), axis=-1)).mean()
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


# todo: apply mlp on xc or not? **
# todo: bidirectional binding? ***
# todo: multi-stream? *
# todo: g one-hot? generalized Gussian **
# todo: apply L1 nom loss on g, x **
# o: g_, x_ must be in the range of [-1, 1]?
# todo: apply downsampling on g?
# todo: x two-hot? *
# todo: tuning: g_dim, n_iteration, view_times, seq_len **
# todo: tuning: lambda yita *
# todo: clamp, fn or tanh? **
# todo: mlp dimension
# todo: balance different losses *
# todo: form which memory? or two memory? *
# todo: form_memory in model2.py
