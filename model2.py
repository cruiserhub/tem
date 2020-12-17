#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:26:32 2020

This is a pytorch implementation of the Tolman-Eichenbaum Machine,

"""
# Standard modules
import numpy as np
import torch
import torch.nn as nn
import pdb
import copy
from scipy.stats import truncnorm
from scipy.special import comb


class Model(torch.nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.hyper = copy.deepcopy(params) # unused
        self.tem = TEM(params=None)
        # todo: self.init_trainable()

    def forward(self, walk, prev_iter=None, prev_M=None):
        steps = self.init_walks(prev_iter)
        for location, x, a in walk:
            if steps is None:
                steps = [self.init_iteration(g, x, [None for _ in range(len(a))], prev_M)]
            x, x_gen, p_inf, p_gen, self.g, g_inf = self.tem(location, x, a)
            # Store this iteration in iteration object in steps list
            steps.append(Iteration(g, x, a, L, M, g_gen, p_gen, x_gen, x_logits, x_inf, g_inf, p_inf))
        # The first step is either a step from a previous walk or initialisiation rubbish, so remove it
        steps = steps[1:]
        return steps

class TEM(nn.Module):
    def __init__(self, params):
        super().__init__()

        # parameters----------------------------
        self.n_action = 4  # todo: 4 or 5
        self.n_stream = 5
        self.hidden_dim = 20
        self.batch_size = 20

        # network dimensions
        self.g_dimension_list = [30, 30, 24, 18, 18]
        self.g_dowonsampled_dimension_list = [10, 10, 8, 6, 6]
        self.p_dimension_list = [100, 100, 80, 60, 60]
        self.xf_dimension_list = [10, 10, 10, 10, 10]
        self.xc_dimension = 10
        self.x_dimension = 45

        # Hebbian parameters
        self.n_iteration = self.n_stream  # 1 iteration for stream 5, while 5 iteration for stream 1
        self.kappa = 0.8
        self.lamda = 0.9999
        self.yita = 0.5
        # -----------------------------------------

        self.state_transition = nn.ModuleList(
            [StateTransition(g_dimension=self.g_dimension_list[i],
                             hidden_dim=self.hidden_dim,
                             n_action=self.n_action)
             for i in range(self.n_stream)]
        )
        self.generator = Generator(
            n_stream=self.n_stream,
            g_downsampled_dimension_list=self.g_dowonsampled_dimension_list,
            p_dimension_list=self.p_dimension_list,
            kappa=self.kappa, n_iteration=self.n_iteration,
            x_dimension=self.x_dimension, hidden_dim=self.hidden_dim,
            xf_dimension_list=self.xf_dimension_list
        )
        self.inference = Inference(
            g_dimension_list=self.g_dimension_list,
            g_downsampled_dimension_list=self.g_dowonsampled_dimension_list,
            hidden_dim=self.hidden_dim, kappa=self.kappa,
            n_iteration=self.n_iteration, n_stream=self.n_stream,
            p_dimension_list=self.p_dimension_list,
            xf_dimension_list=self.xf_dimension_list,
            x_dimension=self.x_dimension,
            xc_dimension=self.xc_dimension
        )
        self.form_memory = FormMemory(
            lamda=self.lamda, yita=self.yita, n_stream=self.n_stream,
            p_dimension_list=self.p_dimension_list,
        )
        self.g = [torch.zeros(self.batch_size, self.g_dimension_list[i])
                  for i in range(self.n_stream)]
        self.register_buffer('M_inf', torch.zeros(self.batch_size,
                    sum(self.p_dimension_list), sum(self.p_dimension_list)))
        self.register_buffer('M_gen', torch.zeros(self.batch_size,
                    sum(self.p_dimension_list), sum(self.p_dimension_list)))

    def forward(self, location, x, a):
        """
        TEM forward
        :param : a: action onehot [batach_size, n_action]
        :param: x: [batch_size, x_dimension]
        :return:
        """
        # todo: multi state transition or shared weights?
        g_mean_path, g_sigma_path =  zip(*[self.state_transition[i](self.g[i], a) for i in range(self.n_stream)])
        g_gen =  [g_mean_path[i] + torch.randn(g_mean_path[i].shape) * g_sigma_path[i]
                  for i in range(self.n_stream)]
        x_gen, p_gen = self.generator(g_gen, self.M_gen)
        p_inf, g_inf, px_t = self.inference(x, g_mean_path, g_sigma_path, self.M_inf)

        self.M_gen = self.form_memory(p_inf, p_gen, self.M_gen)
        self.M_inf = self.form_memory(p_inf, px_t, self.M_inf)
        return x, x_gen, p_inf, p_gen, self.g, g_inf

    def reset(self):
        pass

class Generator(nn.Module):
    """
    Generative model of TEM, varname follows Page e5.
    But State transition is not included, since it's
    shared between Generator a Inference.
    """
    def __init__(self, n_stream, g_downsampled_dimension_list,
                 p_dimension_list, kappa, n_iteration,
                 x_dimension, hidden_dim, xf_dimension_list):
        super().__init__()
        self.n_stream = n_stream
        self.entor2hipp = nn.ModuleList(
            [Entor2Hipp(g_downsampled_dimension=g_downsampled_dimension_list[i],
                        xf_dimension=xf_dimension_list[i])
             for i in range(self.n_stream)]
        )
        self.retrive_memory = RetriveMemory(
            kappa=kappa, n_stream=n_stream, n_iteration=n_iteration,
            p_dimension_list=p_dimension_list
        )
        self.sensory_prediction = nn.ModuleList(
            [SensoryPrediction(g_downsampled_dimension=g_downsampled_dimension_list[i],
                               hidden_dim=hidden_dim,
                               x_dimension=x_dimension,
                               xf_dimension=xf_dimension_list[i]
            )
              for i in range(self.n_stream)]
        )


    def forward(self, g, M):
        """
        Generator forward
        :param g: g after trasition(gt, not gt-1),
            list of tensor, n_stream * [batch_size, g_dimension]
        :param M: Me, tensor, [batch_size, p_dimension, p_dimension]
        :return: x_gen: list of tensor, n_stream * [batch_size, x_dimension]
        :return: p: it's p_gen,
            list of tensor, n_stream * tensor[batch_size, p_dimension]
        """
        # todo: the interaction between different M in the same batch
        g_ = [self.entor2hipp[i](g[i]) for i in range(self.n_stream)]
        p = self.retrive_memory(g_, M)
        x_gen = [self.sensory_prediction[i](p[i]) for i in range(self.n_stream)]
        return x_gen, p


class StateTransition(nn.Module):
    """
    State transition,
    gt = N(.| u = fg(gt-1 + Wa * gt-1), sigma = fg_sigma(gt-1))
    """
    def __init__(self, g_dimension, hidden_dim, n_action):
        super().__init__()
        self.mlp_Wa = MLP(
            in_dim=n_action, out_dim=g_dimension,
            activation=(torch.tanh, None), hidden_dim=hidden_dim,
            bias=(True, False)
        )

        self.fg = nn.Sigmoid()
        self.fg_sigma = nn.Sequential(
            MLP(
                in_dim=g_dimension, out_dim=g_dimension,
                activation=(torch.tanh, torch.exp), hidden_dim=hidden_dim,
                bias=(True, True)
            ),
            nn.Sigmoid())

    def forward(self, g, a):
        """
        gt = N(.| u = fg(gt-1 + Wa * gt-1), sigma = fg_sigma(gt-1))
        :param: g: [batch_size, g_dimension]
        :param: a: [batch_size, n_action]
        :return: g_mean: [batch_size, g_dimension]
        :return: g_sigma: [batch_size, g_dimension]
        """
        # todo: check MLP parameters
        # todo: make sure the activation function of fg / fg_sigma
        g_mean = self.fg(self.mlp_Wa(a) * g + g)    # todo: right?
        g_sigma = self.fg_sigma(g)
        return g_mean, g_sigma

class Entor2Hipp(nn.Module):
    """
    Entorhinal to Hippocampus
    gt_ = W_repeat * f_down(g)
    The module has no trainable parameters.
    """
    def __init__(self, g_downsampled_dimension, xf_dimension):
        super().__init__()
        self.g_downsampled_dimension = g_downsampled_dimension
        # W_repeat [self.g_downsampled_dimension, self.p_dimension]
        self.W_repeat = torch.tensor(np.kron(np.eye(self.g_downsampled_dimension),
                        np.ones((1, xf_dimension))), dtype=torch.float)


    def forward(self, g):
        """
        gt_ = W_repeat * f_down(g)
        :param: g: [batch_size, g_dimension]
        :return: g_ [batch_size, p_dimension]
        """
        # downsample(f_down),
        # g [batch_size, g_dimension] -> [batch_size, g_downsampled_dimension]
        g = g[:, 0: self.g_downsampled_dimension]
        g_ = torch.matmul(g, self.W_repeat)  # g_ [batch_size, p_dimension]
        return g_

class RetriveMemory(nn.Module):
    """
    Rey, reused by both Generative and Inference model.
    Generator: pt ~ N(.| u = attractor(gt_, Mt-1), sigma = f(u))
    Inference: px_t = attractor(x_t, Mt-1)
    """
    def __init__(self, n_stream, n_iteration, p_dimension_list, kappa):
        super().__init__()
        self.n_stream = n_stream
        self.n_iteration = n_iteration
        self.p_dimension_list = p_dimension_list
        self.kappa = kappa

    def forward(self, query, M):
        """
        Generator: pt ~ N(.| u = attractor(gt_, Mt-1), sigma = f(u))
        or Inference: px_t = attractor(xt_, Mt-1)
        :param query: gt_ in Generator or xt_ in Inference
            list, n_stream * (tensor)[batch_size, p_dimension]
        :param M: tensor[batch_size, p_dimension, p_dimension]
        :return: pt(Generator) or px_t(Inference)
            list n_stream * (tensor)[batch_size, p_dimension]
        """
        h_t = torch.cat(query, dim=1)  # [batch_size, sum(p_dimension_list)]
        h_t = self.f_p(h_t)  # todo: whether to use leaky relu?
        # todo: check the dimension of the whole function
        # todo: how to keep M upper bound block matrix
        retrieve_it_mask = [torch.cat((
            torch.ones(sum(self.p_dimension_list[:(self.n_stream-i)])),
            torch.zeros(sum(self.p_dimension_list[(self.n_stream-i):]))
        )) for i in range(self.n_stream)]
        retrieve_it_mask = torch.ones(sum(self.p_dimension_list))
        for tau in range(self.n_iteration):
            h_t = (1 - retrieve_it_mask[tau]) * h_t + retrieve_it_mask[tau] * (
                self.f_p(self.kappa * h_t + torch.squeeze(torch.matmul(torch.unsqueeze(h_t, 1), M))))
        n_p = np.cumsum(np.concatenate(([0], self.p_dimension_list)))
        p = [h_t[:, n_p[f]:n_p[f + 1]] for f in range(self.n_stream)]
        return p
        # todo: optional? for generative model, Gussian sampling is needed,
        #  while for inference model, it's not

    def f_p(self, input):
        return nn.LeakyReLU()(torch.clamp(input, min=-1, max=1))

class SensoryPrediction(nn.Module):
    """
    Sensory Prediction
    xt ~ Cat(fx(pt)), fx(pt) = softmax(fd(wx * W_tile * pt + bx))
    """
    def __init__(self, g_downsampled_dimension, xf_dimension, x_dimension, hidden_dim):
        super().__init__()
        self.mlp_fd = MLP(
            in_dim=xf_dimension, out_dim=x_dimension,
            activation=(nn.functional.elu, None), hidden_dim=hidden_dim,
            bias=(True, True)
        )

        # [W_tile xf_dimension, p_dimension]
        self.W_tile = torch.tensor(np.kron(np.ones((1, g_downsampled_dimension)),
                        np.eye(xf_dimension)), dtype=torch.float)
        self.wx = nn.Parameter(torch.randn(1), requires_grad=True)
        self.bx = nn.Parameter(torch.randn(1), requires_grad=True)
        # todo: Initialization of wx, bx
        self.softmax = nn.Softmax(dim=1)

    def forward(self, p):
        """
        xt ~ Cat(fx(pt)), fx(pt) = softmax(fd(wx * W_tile^T * pt + bx))
        :param p: [batch_size, p_dimension]
        :return: x_gen: [batch_size, x_dimension]
        """
        # todo: make sure if to use wx, bx
        # todo: do sample?
        # todo: make sure the dimension of W_tile is right
        x_gen = self.softmax(self.mlp_fd(self.wx *
                    torch.matmul(p, torch.t(self.W_tile)) + self.bx))
        return x_gen

# ----------------- Decoder -------------------------------

class Inference(nn.Module):
    """
    Inference model of TEM
    TODO: FORM MEMORY
    """
    def __init__(self, n_stream, g_downsampled_dimension_list,
                 xf_dimension_list, p_dimension_list, g_dimension_list,
                 x_dimension, xc_dimension,
                 kappa, n_iteration, hidden_dim
                 ):
        super().__init__()
        self.n_stream = n_stream
        self.compress_sensory = CompressSensory(
            x_dimension=x_dimension,
            xc_dimension=xc_dimension
        )
        self.temporally_filter = nn.ModuleList(
            [TemporallyFilter() for _ in range(self.n_stream)]
        )
        self.sensory2hippo = nn.ModuleList(
            [Sensory2Hippo(
                g_downsampled_dimension=g_downsampled_dimension_list[i],
                xf_dimension=xf_dimension_list[i]
            )
                for i in range(self.n_stream)])
        self.retrieve_memory = RetriveMemory(
            kappa=kappa, n_iteration=n_iteration,
            n_stream=n_stream, p_dimension_list=p_dimension_list
        )
        self.infer_entor = nn.ModuleList(
            [InferEntor(g_dimension=g_dimension_list[i],
                        g_downsampled_dimension=g_downsampled_dimension_list[i],
                        xf_dimension=xf_dimension_list[i],
                        hidden_dim=hidden_dim)
             for i in range(self.n_stream)]
        )
        self.entor2hippo = nn.ModuleList(
            [Entor2Hipp(g_downsampled_dimension=g_downsampled_dimension_list[i],
                        xf_dimension=xf_dimension_list[i])
             for i in range(self.n_stream)]
        )
        self.infer_hippo = [InferHipp()
                            for _ in range(self.n_stream)] # TODO: infer hippocampus

    def forward(self, x, g_mean_path, g_sigma_path, M):
        """
        Inference forward
        :param x: onehot, [batch_size， x_dimension]
        :param g_mean_path: mean of gt(after state transition, not gt-1)
            n_tream * tensor[batch_size, g_dimension]
        :param g_sigma_path: sigma of gt(after state transition, not gt-1)
            n_tream * tensor[batch_size, g_dimension]
        :param M: Mt-1, [batch_size, p, p]
        :return: p: p_inf n_tream * tensor[batch_size, p_dimension]
        :return: g: g_inf n_tream * tensor[batch_size, g_dimension]
        :return: px: px_t used for form memory of inference model
            n_tream * tensor[batch_size, g_dimension]
        """
        xc = self.compress_sensory(x)  # xc [batch_size, xc_dimension]
        xf = [self.temporally_filter[i](xc) for i in range(self.n_stream)] # xf n_stream * [batch_size, xf_dimension]
        x_ = [self.sensory2hippo[i](xf[i]) for i in range(self.n_stream)] # x_ n_stream * [batch_size, p_dimension]
        px = self.retrieve_memory(x_, M)  # px n_stream * [batch_size, p_dimension]
        g = [self.infer_entor[i](px[i], g_mean_path[i], g_sigma_path[i])
             for i in range(self.n_stream)] # n_stream * [batch_size, g_dimension]
        g_ = [self.entor2hippo[i](g[i]) for i in range(self.n_stream)] # g_ n_stream * [batch_size, p_dimension]
        p = [self.infer_hippo[i](g_[i], x_[i]) for i in range(self.n_stream)]  # p n_stream * [batch_size, p_dimension]
        return p, g, px

class CompressSensory(nn.Module):
    """
    Compress Sensory Otion of Inference
    xc = fc(x)
    The module has no trainable parameters.
    """
    def __init__(self, x_dimension, xc_dimension):
        super().__init__()
        self.twohot_table = self.get_twohot_table(
            x_dimension, xc_dimension
        )
        # todo: check twohot table

    def forward(self, x):
        """
        xc = fc(x)
        :param x: [batch_size, x_dimension]
        :return: xc [batch_size, xc_dimension]
        """
        # todo: tobe checked
        xc = torch.stack([self.twohot_table[i]
                        for i in torch.argmax(x, dim=1)], dim=0)
        return xc

    def get_twohot_table(self, x_dimension, xc_dimension):
        twohot_table = [[0] * (xc_dimension - 2) + [1] * 2]
        # We need a compressed code for each possible observation, but it's impossible to have more compressed codes than "n_x_c choose 2"
        for i in range(1, min(int(comb(xc_dimension, 2)), x_dimension)):
            # Copy previous code
            code = twohot_table[-1].copy()
            # Find latest occurrence of [0 1] in that code
            swap = [index for index in range(len(code) - 1, -1, -1) if code[index:index + 2] == [0, 1]][0]
            # Swap those to get new code
            code[swap:swap + 2] = [1, 0]
            # If the first one was swapped: value after swapped pair is 1
            if swap + 2 < len(code) and code[swap + 2] == 1:
                # In that case: move the second 1 all the way back - reverse everything after the swapped pair
                code[swap + 2:] = code[:swap + 1:-1]
            # And append new code to array
            twohot_table.append(code)

        return torch.FloatTensor(twohot_table)

class TemporallyFilter(nn.Module):
    """
    Temporally Filter Sensorium
    xf_t = (1 - alpha_f) * xf_t-1 + alpha_f * xc_t
    """
    def __init__(self):
        super().__init__()
        # todo: initialization of alpha and x_prev
        # todo: check alpha's shape
        self.alpha = nn.Parameter(torch.randn(1), requires_grad=True)
        self.register_buffer('x_prev', torch.zeros(1))

    def forward(self, xc):
        """
        xf_t = (1 - alpha_f) * xf_t-1 + alpha_f * xc_t
        :param xc: [batch_size, xc_dimension]
        :return: xf: [batch_size, xf_dimension]
        """
        xf = self.x_prev * (1 - self.alpha) + self.alpha * xc
        self.x_prev = xf  # attention: deepcopy? detach or not?
        return xf

class Sensory2Hippo(nn.Module):
    """
    Sensory input to Hippocampus
    x_ = W_tile * wp * fn(xf)
    """
    def __init__(self, g_downsampled_dimension, xf_dimension):
        super().__init__()
        self.relu = nn.ReLU()
        # todo: init wp, it's trainable?
        self.wp = nn.Parameter(torch.randn(1), requires_grad=True)
        # [W_tile xf_dimension, p_dimension]
        self.W_tile = torch.tensor(np.kron(np.ones((1, g_downsampled_dimension)),
                                           np.eye(xf_dimension)), dtype=torch.float)

    def forward(self, xf):
        """
        x_ = W_tile * wp * fn(xf)
        :param xf: [batch_size, xf_dimension]
        :return: x_, [batch_size, p_dimension]
        """
        # todo: mean and normalize along which dim?
        # todo: the problem of device
        xf = self.relu(xf - torch.mean(xf, dim=0))
        fn_xf = (xf - torch.mean(xf, dim=0)) / torch.std(xf, dim=0)
        x_ = self.wp * torch.matmul(fn_xf, self.W_tile)
        return x_

class InferEntor(nn.Module):
    """
    Infer Entorhinal
    gt ~  q(gt|gt-1, at-1) * q(gt| px)
    the former item has been calculated by state transition module
    """
    def __init__(self, g_downsampled_dimension, g_dimension, xf_dimension, hidden_dim):
        super().__init__()
        # W_tile [g_downsampled_dimension, p_dimension]
        self.W_repeat = torch.tensor(np.kron(np.eye(g_downsampled_dimension),
                                             np.ones((1, xf_dimension))), dtype=torch.float)
        self.mlp_g_mean_correction = MLP(
            in_dim=g_downsampled_dimension, out_dim=g_dimension,
            activation=(nn.functional.elu, None), hidden_dim=hidden_dim,
            bias=(True, True)
        )
        self.mlp_g_sigma_correction = MLP(
            in_dim=g_downsampled_dimension, out_dim=g_dimension,
            activation=(torch.tanh, torch.exp), hidden_dim=hidden_dim,
            bias=(True, True)
        )

        # todo: refer to inf_g in model.py

    def forward(self, px, g_mean_path, g_sigma_path):
        """
        gt ~  q(gt|gt-1, at-1) * q(gt| px)
        the former item has been calculated by state transition module
        :param px: [batch_size, p_dimension]
        :param: g_mean_path, provided by state transition module
            tensor[batch_size, g_dimension]
        :param: g_sigma_path, provided by state transition module
            tensor[batch_size, g_dimension]
        :return: g tensor[batch_size, g_dimension]
        """
        mlp_input = torch.matmul(px, torch.t(self.W_repeat))
        g_mean_correction = self.mlp_g_mean_correction(mlp_input)
        g_sigma_correction = self.mlp_g_sigma_correction(mlp_input)
        # todo: correction, combine g_mean, g_sigma, g_mean_correction, g_sigma_correction

        # todo: check the combination of path and correction
        # Following, we multiply the two Gussian distribution
        mus = torch.stack([g_mean_correction, g_mean_path], dim=0)
        sigmas = torch.stack([g_sigma_correction, g_sigma_path], dim=0)
        # Calculate inverse variance weighted variance from sum over reciprocal of squared sigmas
        g_sigma = torch.sqrt(torch.sum(1.0 / (sigmas ** 2), dim=0))
        # Calculate inverse variance weighted average
        g_mean = torch.sum(mus / (sigmas ** 2), dim=0) / (g_sigma ** 2)

        g = g_mean  # + torch.randn(g_mean.shape) * g_sigma
        # todo: do sample or not
        return g

class InferHipp(nn.Module):
    """
    Infer Hippocampus
    pt = N(.| u = fp(g_ * x_), sigma = f(x_, g_)
    The module has no trainble parameters
    """
    def __init__(self):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, g_, x_):
        """
        pt = N(.| u = fp(g_ * x_), sigma = f(x_, g_)
        :param g_: [batch_size, p_dimension]
        :param x_: [batch_size, p_dimension]
        :return: p: [batch_size, p_dimension]
        """
        # todo: do sample? the leaky relu slope?
        p = self.leaky_relu(torch.clamp(g_ * x_, min=-1, max=1))
        return p

class FormMemory(nn.Module):
    """
    Form Memory
    Mt = hebbian(Mt-1, pt)
    that is, Mt = lamda * Mt-1 + yita * (pt - pt_) * (pt + pt_)T
    reused by both Generative and Inference model
    The module has no trainable parameters
    """
    def __init__(self, n_stream, p_dimension_list, lamda, yita):
        super().__init__()
        self.n_stream = n_stream
        self.p_dimension_list = p_dimension_list

        self.frequecy_initial = [0.99, 0.3, 0.09, 0.03, 0.01]
        # todo: what's the usage of frequecy_initial
        self.lamda = lamda
        self.yita = yita

        self.M_update_mask = self.get_M_update_mask()
        # todo: check, it should be an upper bound block matrix

    def forward(self, p, p_, M):
        """
        Mt = hebbian(Mt-1, pt)
        Mt = lamda * Mt-1 + yita * (pt - pt_) * (pt + pt_)T
        :param p: pt, list, n_stream * [batch_size, p_dimension]
        :param p_: pt_, list, n_stream * [batch_size, p_dimension]
        :param M: Mt-1, [batch_size, p_dimension, p_dimension]
        :return: M: Mt, [batch_size, p_dimension, p_dimension]
        """
        p = torch.cat(p, dim=1)  # [batch_size, sum(p_dimension_list)]
        p_ = torch.cat(p_, dim=1)  # [batch_size, sum(p_dimension_list)]
        M_new = torch.squeeze(torch.matmul(torch.unsqueeze(p + p_, 2),
                                           torch.unsqueeze(p - p_, 1)))
        M_new = M_new * self.M_update_mask
        # todo: what' s ovc in parameters.py
        M = self.lamda * M + self.yita * M_new
        return M

    def get_M_update_mask(self):
        M_update_mask = torch.zeros((np.sum(self.p_dimension_list),
                                     np.sum(self.p_dimension_list)), dtype=torch.float)
        n_p = np.cumsum(np.concatenate(([0], self.p_dimension_list)))
        # Entry M_ij (row i, col j) is the connection FROM cell i TO cell j. Memory is retrieved by h_t+1 = h_t * M, i.e. h_t+1_j = sum_i {connection from i to j * h_t_i}
        for f_from in range(self.n_stream):
            for f_to in range(self.n_stream):
                # For connections that involve separate object vector modules: these are connected to all normal modules, but hierarchically between object vector modules
                if f_from > self.n_stream or f_to > self.n_stream:
                    # If this is a connection between object vector modules: only allow for connection from low to high frequency
                    if (f_from > self.n_stream and f_to > self.n_stream):
                        if self.frequecy_initial[f_from] <= self.frequecy_initial[f_to]:
                            M_update_mask[n_p[f_from]:n_p[f_from + 1], n_p[f_to]:n_p[f_to + 1]] = 1.0
                    # If this is a connection to between object vector and normal modules: allow any connections, in both directions
                    else:
                        M_update_mask[n_p[f_from]:n_p[f_from + 1], n_p[f_to]:n_p[f_to + 1]] = 1.0
                # Else: this is a connection between abstract location frequency modules; only allow for connections if it goes from low to high frequency
                else:
                    if self.frequecy_initial[f_from] <= self.frequecy_initial[f_to]:
                        M_update_mask[n_p[f_from]:n_p[f_from + 1], n_p[f_to]:n_p[f_to + 1]] = 1.0
        return M_update_mask


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, activation, hidden_dim, bias):
        # First call super class init function to set up torch.nn.Module style model and inherit it's functionality
        super(MLP, self).__init__()
        # Check if this network consists of module: are input and output dimensions lists? If not, make them (but remember it wasn't)
        if type(in_dim) is list:
            self.is_list = True
        else:
            in_dim = [in_dim]
            out_dim = [out_dim]
            self.is_list = False
        # Find number of modules
        self.N = len(in_dim)
        # Create weights (input->hidden, hidden->output) for each module
        self.w = torch.nn.ModuleList([])
        for n in range(self.N):
            # If number of hidden dimensions is not specified: mean of input and output
            if hidden_dim is None:
                hidden = int(np.mean([in_dim[n], out_dim[n]]))
            else:
                hidden = hidden_dim[n] if self.is_list else hidden_dim
                # Each module has two sets of weights: input->hidden and hidden->output
            self.w.append(torch.nn.ModuleList(
                [torch.nn.Linear(in_dim[n], hidden, bias=bias[0]), torch.nn.Linear(hidden, out_dim[n], bias=bias[1])]))
        # Copy activation function for hidden layer and output layer
        self.activation = activation
        # Initialise all weights
        with torch.no_grad():
            for from_layer in range(2):
                for n in range(self.N):
                    # Set weights to xavier initalisation
                    torch.nn.init.xavier_normal_(self.w[n][from_layer].weight)
                    # Set biases to 0
                    if bias[from_layer]:
                        self.w[n][from_layer].bias.fill_(0.0)

    def set_weights(self, from_layer, value):
        # If single value is provided: copy it for each module
        if type(value) is not list:
            input_value = [value for n in range(self.N)]
        else:
            input_value = value
        # Run through all modules and set weights starting from requested layer to the specified value
        with torch.no_grad():
            # MLP is setup as follows: w[module][layer] is Linear object, w[module][layer].weight is Parameter object for linear weights, w[module][layer].weight.data is tensor of weight values
            for n in range(self.N):
                # If a tensor is provided: copy the tensor to the weights
                if type(input_value[n]) is torch.Tensor:
                    self.w[n][from_layer].weight.copy_(input_value[n])
                    # If only a single value is provided: set that value everywhere
                else:
                    self.w[n][from_layer].weight.fill_(input_value[n])

    def forward(self, data):
        # Make input data into list, if this network doesn't consist of modules
        if self.is_list:
            input_data = data
        else:
            input_data = [data]
        # Run input through network for each module
        output = []
        for n in range(self.N):
            # Pass through first weights from input to hidden layer
            module_output = self.w[n][0](input_data[n])
            # Apply hidden layer activation
            if self.activation[0] is not None:
                module_output = self.activation[0](module_output)
            # Pass through second weights from hidden to output layer
            module_output = self.w[n][1](module_output)
            # Apply output layer activation
            if self.activation[1] is not None:
                module_output = self.activation[1](module_output)
            # Transpose output again to go back to column vectors instead of row vectors
            output.append(module_output)
            # If this network doesn't consist of modules: select output from first module to return
        if not self.is_list:
            output = output[0]
        # And return output
        return output

# TODO: MLP PARAMeters tuning: activation, bias, hidden

if __name__ == '__main__':
    # import random
    # tem = TEM(params=None)
    # x_dimension = 45
    # n_action = 4
    # batch_size = 20
    # location = None
    # x = torch.stack([torch.eye(x_dimension)[random.randint(0, x_dimension - 1)]
    #                  for _ in range(batch_size)], dim=0)
    # a = torch.stack([torch.eye(n_action)[random.randint(0, n_action - 1)]
    #                  for _ in range(batch_size)], dim=0)
    # tem(None, x, a)
    #
    # g = torch.randn()

    import random

    ng = 20
    nxf = 20
    form_memory =  FormMemory(1, [ng * nxf], lamda=0.9999, yita=0.5)
    retrive_memory = RetriveMemory(1, 20, [ng*nxf], kappa=0.8)
    # [W_tile xf_dimension, p_dimension]
    W_tile = torch.tensor(np.kron(np.ones((1, nxf)),
                                       np.eye(ng)), dtype=torch.float)

    from scipy.stats import ortho_group
    A = np.float32(ortho_group.rvs(dim=ng))
    print(np.max(A), np.min(A))
    A = torch.FloatTensor(A)
    B = np.float32(ortho_group.rvs(dim=nxf))
    B = torch.FloatTensor(B)

    T = 10
    # g = [torch.randn(1, ng) for _ in range(T)]
    g = [A[i].reshape(1, ng)
          for i in range(T)]
    # xf = [torch.randn(1, nxf) for _ in range(T)]
    xf = [B[i].reshape(1, nxf)
         for i in range(T)]

    M = torch.zeros(1, ng*nxf, ng*nxf)
    for iter in range(100):
        for t in range(T):
            p = [(torch.t(xf[t]) * g[t]).reshape(1, ng*nxf)]
            p_ = retrive_memory([torch.matmul(g[t], W_tile)], M)
            M = form_memory(p, p_, M)
            print(torch.sum(torch.abs(p[0] - p_[0])))
            # print(p[0], p_[0])
        print('-------------')



