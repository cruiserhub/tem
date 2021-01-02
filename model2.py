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
        self.hyper = copy.deepcopy(params) # todo: lambda and eta should be updated
        self.n_actions = 4  # all action are 4 + 1 (static)
        self.n_stream = 5
        self.batch_size = 16  # also a batch_size param in class TEM
        self.device = 'cuda:0'
        self.tem = TEM(params=None,
                       n_action=self.n_actions,
                       n_stream=self.n_stream,
                       batch_size=self.batch_size,
                       device=self.device).to(self.device)
        # tuning: self.init_trainable()

    def forward(self, walk, prev_iter=None, prev_M=None):
        self.tem.detach_buffer()
        # todo: tem yita and lamda updated
        self.tem.yita = self.hyper['eta']
        self.tem.lamda = self.hyper['lambda']
        self.tem.form_memory.lamda = self.hyper['lambda']
        self.tem.form_memory.yita = self.hyper['eta']

        # todo: tem detach buffer should be after loss.backward()
        if prev_iter is not None:
            self.tem.reset(prev_iter[0].a)
            a_prev = prev_iter[0].a
        else:
            # todo: this reset seems unnecessary
            # self.tem.reset()
            # todo: a_prev changed from 0 to None
            a_prev = [None for _ in range(self.batch_size)]
        steps = []
        for location, x, a in walk:
            x = x.clone().detach().to(self.device)

            x_gen_logits, p_inf, p1, g_inf, g_gen, p_inf_x, (x_gen_logits_3, g4) = self.tem(location, x, a_prev)
            loss = self.calculate_loss(x, x_gen_logits, p_inf, p1, g_inf, g_gen, p_inf_x, x_gen_logits_3, g4)
            # Store this iteration in iteration object in steps list
            steps.append(Iteration(location, x, loss, x_gen_logits, a))
            a_prev = a
        return steps

    def calculate_loss(self, x, x_gen_logits, p_inf, p1, g_inf, g_gen, p_inf_x, x_gen_logits_3, g4):
        # todo: check losses
        # attention: Here p1 is p_gen in model.py, it's retrieved from g_inf
        labels = torch.argmax(x, dim=-1).unsqueeze(1)

        # tuning: loss_x_gen, loss_x_g, loss_x_p is calculated from only stream 0 now
        loss_x_gen = torch.stack([
            nn.CrossEntropyLoss()(x_gen_logits[0][0][j].unsqueeze(0), labels[j])
            for j in range(self.batch_size)]).to('cpu')

        loss_x_g = torch.stack([
            nn.CrossEntropyLoss()(x_gen_logits[1][0][j].unsqueeze(0), labels[j])
            for j in range(self.batch_size)]).to('cpu')

        loss_x_p = torch.stack([
            nn.CrossEntropyLoss()(x_gen_logits[2][0][j].unsqueeze(0), labels[j])
            for j in range(self.batch_size)]).to('cpu')

        # loss_p_x = 0.5 * torch.stack([
        #     sum([nn.MSELoss()(p_inf[i][j], p_inf_x[i][j])
        #           for i in range(self.n_stream)])
        #     for j in range(self.batch_size)]).to('cpu')
        # todo: the calculation of mse_loss has been updated according to model.py
        loss_p_x = torch.sum(torch.stack([0.5 * torch.sum(torch.nn.MSELoss(reduction='none')
            (p_inf[i], p_inf_x[i]), dim=-1) for i in range(self.n_stream)], dim=0), dim=0).to('cpu')

        loss_p = torch.sum(torch.stack([0.5 * torch.sum(torch.nn.MSELoss(reduction='none')
            (p_inf[i], p1[i]), dim=-1) for i in range(self.n_stream)], dim=0), dim=0).to('cpu')

        loss_g = torch.sum(torch.stack([0.5 * torch.sum(torch.nn.MSELoss(reduction='none')
            (g_inf[i], g_gen[i]), dim=-1) for i in range(self.n_stream)], dim=0), dim=0).to('cpu')

        loss_reg_g = torch.sum(torch.stack([
            torch.sum(g ** 2, dim=1) for g in g_inf], dim=0), dim=0).to('cpu')

        loss_reg_p = torch.sum(torch.stack([
            torch.sum(torch.abs(p), dim=1) for p in p_inf], dim=0), dim=0).to('cpu')
        # import random
        # if random.random() > 0.99:
        #     print('------------------')
        #     print(p_inf_x[1].abs().mean().item(), p_inf[1].abs().mean().item(),
        #           g_inf[1].abs().mean().item(), g_gen[1].abs().mean().item())
        #     print([torch.max(g).item() for g in g_gen], [torch.min(g).item() for g in g_gen])
        #     print([torch.max(g).item() for g in g_inf], [torch.min(g).item() for g in g_inf])
        #     print([torch.max(p).item() for p in p_inf], [torch.min(p).item() for p in p_inf])
        #     print([torch.max(p).item() for p in p_inf_x], [torch.min(p).item() for p in p_inf_x])
        loss_x3 = torch.stack([
            nn.CrossEntropyLoss()(x_gen_logits_3[0][j].unsqueeze(0), labels[j])
            for j in range(self.batch_size)]).to('cpu')
        loss_g4 = torch.stack([
            sum([nn.MSELoss()(g4[i][j], g_gen[i][j])
                  for i in range(self.n_stream)])
            for j in range(self.batch_size)]).to('cpu')

        loss_padding = torch.zeros(16).to('cpu')
        return [loss_p, loss_p_x, loss_x_gen, loss_x_g,
                loss_x_p, loss_g,
                loss_reg_g, loss_reg_p, loss_x3, loss_g4]
    # todo: using loss_reg_g, loss_reg_p

class TEM(nn.Module):
    def __init__(self, params, n_action, n_stream, batch_size, device):
        super().__init__()

        # parameters----------------------------
        self.n_action = n_action  # 4 action and 1 static action additionally
        self.n_stream = n_stream
        self.batch_size = batch_size
        self.device = device

        # network dimensions
        self.g_dimension_list = [30, 30, 24, 18, 18]
        self.g_downsampled_dimension_list = [10, 10, 8, 6, 6]
        self.p_dimension_list = [100, 100, 80, 60, 60]
        self.xf_dimension_list = [10, 10, 10, 10, 10]
        self.xc_dimension = 10
        self.x_dimension = 45
        self.hidden_dim = 20    # MLP hidden_dim
        # todo: apply initialization to alpha
        self.initial_frequency = [0.99, 0.3, 0.09, 0.03, 0.01]  # used to init alpha

        # Hebbian parameters
        self.n_iteration = self.n_stream  # 1 iteration for stream 5, while 5 iteration for stream 1
        self.kappa = 0.8
        self.lamda = 0.9999
        self.yita = 0.5
        # --------------------------------------

        self.state_transition = nn.ModuleList(
            [StateTransition(g_dimension=self.g_dimension_list[i],
                             hidden_dim=self.hidden_dim,
                             n_action=self.n_action,
                             device=self.device)
             for i in range(self.n_stream)]
        )
        self.generator = Generator(
            n_stream=self.n_stream,
            g_downsampled_dimension_list=self.g_downsampled_dimension_list,
            p_dimension_list=self.p_dimension_list,
            kappa=self.kappa, n_iteration=self.n_iteration,
            x_dimension=self.x_dimension, hidden_dim=self.hidden_dim,
            xf_dimension_list=self.xf_dimension_list,
            device=device,
            g_dimension_list=self.g_dimension_list
        )
        self.inference = Inference(
            g_dimension_list=self.g_dimension_list,
            g_downsampled_dimension_list=self.g_downsampled_dimension_list,
            hidden_dim=self.hidden_dim, kappa=self.kappa,
            n_iteration=self.n_iteration, n_stream=self.n_stream,
            p_dimension_list=self.p_dimension_list,
            xf_dimension_list=self.xf_dimension_list,
            x_dimension=self.x_dimension,
            xc_dimension=self.xc_dimension,
            device=device
        )
        self.form_memory = FormMemory(
            lamda=self.lamda, yita=self.yita, n_stream=self.n_stream,
            p_dimension_list=self.p_dimension_list,
        )
        # g_init std set to 0, four places
        self.g = [0 * torch.clamp(torch.randn(self.batch_size,
                                                self.g_dimension_list[i]), -2, 2).to(device)
                  for i in range(self.n_stream)]
        self.register_buffer('M_inf', torch.zeros(self.batch_size,
                    sum(self.p_dimension_list), sum(self.p_dimension_list)))
        self.register_buffer('M_gen', torch.zeros(self.batch_size,
                    sum(self.p_dimension_list), sum(self.p_dimension_list)))

        # -------------------models for additional loss-----------
        self.entor2hipp1 = nn.ModuleList(
            [Entor2Hipp(g_downsampled_dimension=self.g_downsampled_dimension_list[i],
                        xf_dimension=self.xf_dimension_list[i],
                        g_dimension=self.g_dimension_list[i])
             for i in range(self.n_stream)]
        )
        self.retrive_memory1 = RetriveMemory(
            kappa=self.kappa, n_stream=self.n_stream, n_iteration=self.n_iteration,
            p_dimension_list=self.p_dimension_list,
            retrive_it_mask=True
        )

        # ---------------modules for additional observation-----
        self.retrive_memory3 = RetriveMemory(
            kappa=self.kappa, n_stream=self.n_stream, n_iteration=self.n_iteration,
            p_dimension_list=self.p_dimension_list,
            retrive_it_mask=True
        )
        self.sensory_prediction3 = nn.ModuleList(
            [SensoryPrediction(g_downsampled_dimension=self.g_downsampled_dimension_list[i],
                               hidden_dim=self.hidden_dim,
                               x_dimension=self.x_dimension,
                               xf_dimension=self.xf_dimension_list[i]
                               )
             for i in range(self.n_stream)]
        )
        self.retrive_memory4 = RetriveMemory(
            kappa=self.kappa, n_stream=self.n_stream, n_iteration=self.n_iteration,
            p_dimension_list=self.p_dimension_list,
            retrive_it_mask=True
        )

    def forward(self, location, x, a):
        """
        TEM forward
        :param : a: list, len(a) batch_size, None means beginning
        :param: x: [batch_size, x_dimension]
        :return:
        """
        # tuning: multi state transition or shared weights?
        g_mean_path, g_sigma_path =  zip(*[self.state_transition[i](self.g[i], a) for i in range(self.n_stream)])
        # g_gen =  [g_mean_path[i] + torch.randn(g_mean_path[i].shape).to(self.device) * g_sigma_path[i]
        #           for i in range(self.n_stream)]
        # tuning: need to sample?
        g_gen = g_mean_path

        x_gen_logits, p_gen = self.generator(g_gen, self.M_gen)
        p_inf, g_inf, px_t = self.inference(x, g_mean_path, g_sigma_path, self.M_inf)

        self.g = g_inf  # attention: it's shallow copy
        # self.M_gen = self.form_memory(p_inf, p_gen, self.M_gen)
        # todo; M_gen form has been modified
        self.M_inf = self.form_memory(p_inf, px_t, self.M_inf, do_hierarchical_connections=False)
        # todo: ----------------------------------------
        #  1. g_inf->Entor2Hipp->retrive_memory->SensoryPrediction
        g_1 = [self.entor2hipp1[i](g_inf[i]) for i in range(self.n_stream)]
        p1 = self.retrive_memory1(g_1, self.M_gen)
        self.M_gen = self.form_memory(p_inf, p1, self.M_gen, do_hierarchical_connections=True)
        x_gen_logits_1 = [self.generator.sensory_prediction[i](p1[i]) for i in range(self.n_stream)]
        # 2. gen x from p_inf->SensoryPrediction
        x_gen_logits_2 = [self.generator.sensory_prediction[i](p_inf[i]) for i in range(self.n_stream)]
        # todo: 3 sensory_predictions are identical
        # ---------------------------------------------
        # 3. p_x -> retrieve_memory -> sensoryprediction -> x
        p3 = self.retrive_memory3(px_t, self.M_inf)
        x_gen_logits_3 = [self.sensory_prediction3[i](p3[i]) for i in range(self.n_stream)]
        # 4. p_gen -> retrieve_memory -> infer_hippo -> g
        p4 = self.retrive_memory4(p_gen, self.M_gen)
        g4 = [self.inference.infer_entor[i](p4[i], g_mean_path[i], g_sigma_path[i])
         for i in range(self.n_stream)]
        # import random
        # if random.random() > 0.99:
        #     print(self.M_inf[0].mean(), self.M_inf[0].std())
        #     print(torch.abs(px_t[0][0]).mean(), px_t[0][0].std())
        #     print('-----------------------')
        return (x_gen_logits, x_gen_logits_1, x_gen_logits_2), \
               p_inf, p1, g_inf, g_gen, px_t, (x_gen_logits_3, g4)

    def reset(self, prev_a=None):
        """

        :param prev_a: prev_a item is None indicate that it's a new walk
            list, len(prev_a) = batch_size
        :return:
        """
        # TODO: why are there so many copies of M ,g, x
        if prev_a is None:
            self.M_inf = torch.zeros(self.batch_size,
                    sum(self.p_dimension_list), sum(self.p_dimension_list)).to(self.device)
            self.M_gen = torch.zeros(self.batch_size,
                    sum(self.p_dimension_list), sum(self.p_dimension_list)).to(self.device)
            for s in range(self.n_stream):
                # todo: g init changed, but clamp may be different from trucnorm
                self.g[s] = 0 * torch.clamp(torch.randn(self.batch_size,
                            self.g_dimension_list[s]), -2, 2).to(self.device)
                self.inference.temporally_filter[s].x_prev = torch.zeros(1).to(self.device)
        else:
            for i, a in enumerate(prev_a):
                if a is None:
                    self.M_inf[i] = 0
                    self.M_gen[i] = 0
                    for s in range(self.n_stream):
                        self.g[s][i] = 0 * torch.clamp(
                                    torch.randn(self.g_dimension_list[s]), -2, 2).to(self.device)
                        self.inference.temporally_filter[s].x_prev[i] = 0

        # todo: an error in reset has been revised, M, g and x are wrongly reset frequently
        # todo: now init g with randn instead of zeros

    def detach_buffer(self):
        self.M_inf = self.M_inf.detach().clone()
        self.M_gen = self.M_gen.detach().clone()
        for s in range(self.n_stream):
            self.g[s] = self.g[s].detach().clone()
            self.inference.temporally_filter[s].x_prev \
                = self.inference.temporally_filter[s].x_prev.detach().clone()


class Generator(nn.Module):
    """
    Generative model of TEM, varname follows Page e5.
    But State transition is not included, since it's
    shared between Generator a Inference.
    """
    def __init__(self, n_stream, g_downsampled_dimension_list,
                 p_dimension_list, kappa, n_iteration,
                 x_dimension, hidden_dim, xf_dimension_list, device, g_dimension_list):
        super().__init__()
        self.n_stream = n_stream
        self.entor2hipp = nn.ModuleList(
            [Entor2Hipp(g_downsampled_dimension=g_downsampled_dimension_list[i],
                        xf_dimension=xf_dimension_list[i],
                        g_dimension=g_dimension_list[i])
             for i in range(self.n_stream)]
        )
        self.retrive_memory = RetriveMemory(
            kappa=kappa, n_stream=n_stream, n_iteration=n_iteration,
            p_dimension_list=p_dimension_list,
            retrive_it_mask=True
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
        :return: x_gen_logits: list of tensor, n_stream * [batch_size, x_dimension]
        :return: p: it's p_gen,
            list of tensor, n_stream * tensor[batch_size, p_dimension]
        """

        g_ = [self.entor2hipp[i](g[i]) for i in range(self.n_stream)]
        p = self.retrive_memory(g_, M)
        x_gen_logits = [self.sensory_prediction[i](p[i]) for i in range(self.n_stream)]
        return x_gen_logits, p


class StateTransition(nn.Module):
    # todo: asymmetry in state_transition(g_connection)
    """
    State transition,
    gt = N(.| u = fg(gt-1 + Wa * gt-1), sigma = fg_sigma(gt-1))
    """
    def __init__(self, g_dimension, hidden_dim, n_action, device):
        super().__init__()
        self.g_dimension = g_dimension
        self.n_action = n_action
        self.device = device

        self.mlp_Wa = MLP(
            in_dim=n_action, out_dim=g_dimension * g_dimension,
            activation=(torch.tanh, None), hidden_dim=hidden_dim,
            bias=(True, False)
        )

        self.fg = nn.Sigmoid()
        self.fg_sigma = nn.Sequential(
            MLP(
                in_dim=g_dimension, out_dim=g_dimension,
                activation=(torch.tanh, torch.exp), hidden_dim=2 * g_dimension,
                bias=(True, True)
            ),
            nn.Identity())
            # nn.Sigmoid())

    def forward(self, g, a):
        """
        gt = N(.| u = fg(gt-1 + Wa * gt-1), sigma = fg_sigma(gt-1))
        :param: g: [batch_size, g_dimension]
        :param: a: list, len(a) = batch_size, None begins a walk
        :return: g_mean: [batch_size, g_dimension]
        :retur_sigma: [batch_size, g_dimension]
        """
        a_list = a.copy()
        a = [ai if ai is not None else 0 for ai in a]
        a = torch.zeros((len(a), self.n_action)).scatter_(
            1, torch.clamp(torch.tensor(a).unsqueeze(1) - 1, min=0),
            1.0 * (torch.tensor(a).unsqueeze(1) > 0)).to(self.device)

        g_mean = (torch.matmul(g.unsqueeze(1),
            self.mlp_Wa(a).view(-1, self.g_dimension, self.g_dimension)).squeeze() + g)
        # tuning: fg is sigmoid here but clamp in model.py
        g_mean = torch.clamp(g_mean, min=-1, max=1)
        g_sigma = self.fg_sigma(g)  # tuning: no sigmoid in model.py
        for i, ai in enumerate(a_list):
            # todo: if ai is None, re-init g_mean, g_sigma,
            if ai is None:
                g_mean[i] = 0 * torch.clamp(torch.randn(*g_mean[i].shape), -2, 2)
                g_sigma[i] = 1  # todo: modify
        # todo: list a has been propagated to state_transition
        #  to inform the beginning of a sequence and init g_mean, g_sigma
        return g_mean, g_sigma

class Entor2Hipp(nn.Module):
    """
    Entorhinal to Hippocampus
    gt_ = W_repeat * f_down(g)
    The module has no trainable parameters.
    """
    def __init__(self, g_downsampled_dimension, xf_dimension, g_dimension):
        super().__init__()
        self.g_downsampled_dimension = g_downsampled_dimension
        self.g_dimension = g_dimension
        # W_repeat [self.g_downsampled_dimension, self.p_dimension]
        self.W_repeat = nn.Parameter(torch.tensor(np.kron(np.eye(self.g_downsampled_dimension),
                        np.ones((1, xf_dimension))), dtype=torch.float), requires_grad=False)
        self.W_g_downsample = nn.Parameter(torch.cat([torch.eye(g_downsampled_dimension, dtype=torch.float),
                                         torch.zeros((g_dimension-g_downsampled_dimension, g_downsampled_dimension), dtype=torch.float)]),
                                           requires_grad=False)

    def forward(self, g):
        """
        gt_ = W_repeat * f_down(g)
        :param: g: [batch_size, g_dimension]
        :return: g_ [batch_size, p_dimension]
        """
        # downsample(f_down),
        # g [batch_size, g_dimension] -> [batch_size, g_downsampled_dimension]
        # g = g[:, 0: self.g_downsampled_dimension].clone()
        # todo: add W_G_downsample
        g = torch.matmul(g, self.W_g_downsample)
        g_ = torch.matmul(g, self.W_repeat)  # g_ [batch_size, p_dimension]
        return g_

class RetriveMemory(nn.Module):
    """
    Rey, reused by both Generative and Inference model.
    Generator: pt ~ N(.| u = attractor(gt_, Mt-1), sigma = f(u))
    Inference: px_t = attractor(x_t, Mt-1)
    """
    def __init__(self, n_stream, n_iteration, p_dimension_list, kappa, retrive_it_mask):
        super().__init__()
        self.n_stream = n_stream
        self.n_iteration = n_iteration
        self.p_dimension_list = p_dimension_list
        self.kappa = kappa
        if retrive_it_mask:
            self.retrieve_it_mask = nn.ParameterList([nn.Parameter(torch.cat((
                torch.ones(sum(self.p_dimension_list[:(self.n_stream - i)])),
                torch.zeros(sum(self.p_dimension_list[(self.n_stream - i):]))
            )), requires_grad=False) for i in range(self.n_iteration)])
        else:
            self.retrieve_it_mask = nn.ParameterList([
                nn.Parameter(torch.ones(sum(self.p_dimension_list)),requires_grad=False)
                for _ in range(self.n_iteration)
            ])
        # todo: mask_inf is composed of ones, that's to say, inference doesn't need mask

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
        h_t = self.f_p(h_t)  # tuning: whether to use leaky relu?
        # todo: additional h_t is multiplied
        for tau in range(self.n_iteration):
            h_t = (1 - self.retrieve_it_mask[tau]) * h_t + self.retrieve_it_mask[tau] * (
                self.f_p(self.kappa * h_t + h_t * torch.squeeze(torch.matmul(torch.unsqueeze(h_t, 1), M))))
        n_p = np.cumsum(np.concatenate(([0], self.p_dimension_list)))
        p = [h_t[:, n_p[f]:n_p[f + 1]] for f in range(self.n_stream)]
        return p
        # tuning: optional? for generative model, Gussian sampling is needed,
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
            activation=(nn.functional.elu, None), hidden_dim=20 * xf_dimension,
            bias=(True, True)
        )
        # tuning: 20 * xf_dimension in model.py, also seem too large

        # [W_tile xf_dimension, p_dimension]
        self.W_tile = nn.Parameter(torch.tensor(np.kron(np.ones((1, g_downsampled_dimension)),
                        np.eye(xf_dimension)), dtype=torch.float), requires_grad=False)
        self.wx = nn.Parameter(torch.ones(1), requires_grad=True)
        self.bx = nn.Parameter(torch.zeros(1), requires_grad=True)
        # tuning: Initialization of wx, bx, wx changed from randn to ones

    def forward(self, p):
        """
        xt ~ Cat(fx(pt)), fx(pt) = softmax(fd(wx * W_tile^T * pt + bx))
        But we didn't do softmax here, just return logits before softmax
        :param p: [batch_size, p_dimension]
        :return: x_gen_logits: [batch_size, x_dimension]
        """
        x_gen_logits = self.mlp_fd(self.wx *
                    torch.matmul(p, torch.t(self.W_tile)) + self.bx)
        return x_gen_logits

# ----------------- Decoder -------------------------------

class Inference(nn.Module):
    """
    Inference model of TEM
    """
    def __init__(self, n_stream, g_downsampled_dimension_list,
                 xf_dimension_list, p_dimension_list, g_dimension_list,
                 x_dimension, xc_dimension,
                 kappa, n_iteration, hidden_dim, device
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
            n_stream=n_stream, p_dimension_list=p_dimension_list,
            retrive_it_mask=False
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
                        xf_dimension=xf_dimension_list[i],
                        g_dimension=g_dimension_list[i])
             for i in range(self.n_stream)]
        )
        self.infer_hippo = [InferHipp()
                            for _ in range(self.n_stream)]

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
        self.twohot_table = nn.Parameter(self.get_twohot_table(
            x_dimension, xc_dimension
        ), requires_grad=False)

    def forward(self, x):
        """
        xc = fc(x)
        :param x: [batch_size, x_dimension]
        :return: xc [batch_size, xc_dimension]
        """
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
        # tuning: initialization of alpha, changed from randn to ones
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.register_buffer('x_prev', torch.zeros(1))

    def forward(self, xc):
        """
        xf_t = (1 - alpha_f) * xf_t-1 + alpha_f * xc_t
        :param xc: [batch_size, xc_dimension]
        :return: xf: [batch_size, xf_dimension]
        """
        alpha = nn.Sigmoid()(self.alpha)  # scale alpha to 0~1
        xf = self.x_prev * (1 - alpha) + alpha * xc
        # todo: xf.clone() -> xf
        self.x_prev = xf
        return xf

class Sensory2Hippo(nn.Module):
    """
    Sensory input to Hippocampus
    x_ = W_tile * wp * fn(xf)
    """
    def __init__(self, g_downsampled_dimension, xf_dimension):
        super().__init__()
        self.relu = nn.ReLU()
        # tuning: init wp, changed from randn to ones
        self.wp = nn.Parameter(torch.ones(1), requires_grad=True)
        # [W_tile xf_dimension, p_dimension]
        self.W_tile = nn.Parameter(torch.tensor(np.kron(np.ones((1, g_downsampled_dimension)),
                                           np.eye(xf_dimension)), dtype=torch.float),
                                   requires_grad=False)

    def forward(self, xf):
        """
        x_ = W_tile * wp * fn(xf)
        :param xf: [batch_size, xf_dimension]
        :return: x_, [batch_size, p_dimension]
        """
        # todo: mean and normalize along which dim?
        # tuning: model.py apply sigmoid on x_
        xf = self.relu(xf - torch.mean(xf))
        fn_xf = torch.nn.functional.normalize(xf, p=2, dim=-1)
        # fn_xf = (xf - torch.mean(xf, dim=-1).unsqueeze(1)) \
        #         / torch.std(xf, dim=-1).unsqueeze(1)
        # todo: sigmoid is applied in self.wp
        x_ = nn.Sigmoid()(self.wp) * torch.matmul(fn_xf, self.W_tile)
        return x_
        # todo: fn has been set to normalize(decide by norm)

class InferEntor(nn.Module):
    """
    Infer Entorhinal
    gt ~  q(gt|gt-1, at-1) * q(gt| px)
    the former item has been calculated by state transition module
    """
    def __init__(self, g_downsampled_dimension, g_dimension, xf_dimension, hidden_dim):
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
        # todo: it's different from model.py
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
        # todo: add clamp to g_mean_correction
        g_mean_correction = torch.clamp(g_mean_correction, -1, 1)
        g_sigma_correction = self.mlp_g_sigma_correction(mlp_input)
        g = (10 * g_mean_path + g_mean_correction) / 11
        return g
        # todo: simply average over g_mean_path and g_mean_correction

        # todo: utils.py inv_var_weight fun is wrong in g_sigma, luckily, it didn't do sample
        # Following, we multiply the two Gussian distribution
        mus = torch.stack([g_mean_correction, g_mean_path], dim=0)
        sigmas = torch.stack([g_sigma_correction, g_sigma_path], dim=0)
        g_sigma = 1 / torch.sqrt(torch.sum(1.0 / (sigmas ** 2), dim=0))
        g_mean = torch.sum(mus / (sigmas ** 2), dim=0) * (g_sigma ** 2)

        g = g_mean  # + torch.randn(g_mean.shape) * g_sigma
        # tuning: do sample or not
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
        # tuning: do sample? the leaky relu slope?
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
        self.lamda = lamda
        self.yita = yita

        self.M_update_mask = nn.Parameter(self.get_M_update_mask(),
                                          requires_grad=False)
        # todo: check, it should be an upper bound block matrix, but it's down bound

    def forward(self, p, p_, M, do_hierarchical_connections=True):
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
        # todo: add do_hierarchical_connections, inf False, gen True
        if do_hierarchical_connections:
            M_new = M_new * self.M_update_mask
        # todo: clamp M
        M = torch.clamp(self.lamda * M + self.yita * M_new, -1, 1)
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
    def __init__(self, in_dim, out_dim, activation=(torch.nn.functional.elu, None), hidden_dim=None, bias=(True, True)):
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


class Iteration:
    def __init__(self, g=None, x=None, L=None,  x_gen_logits=None, a=None):
        # Copy all inputs
        self.g = g   # location
        self.x = x    # x
        self.L = L   # l
        self.a = a
        self.x_gen_logits = [x[0] for x in x_gen_logits] # x_gen_logits
        self.M = [torch.zeros(1), torch.zeros(1)]

    def correct(self):
        # Detach observation and all predictions
        observation = self.x.cpu().detach().numpy()
        predictions = [tensor.cpu().detach().numpy() for tensor in self.x_gen_logits]
        # Did the model predict the right observation in this iteration?
        accuracy = [np.argmax(prediction, axis=-1) == np.argmax(observation, axis=-1) for prediction in predictions]
        return accuracy

    def detach(self):
        # Detach all tensors contained in this iteration
        # todo: detach seems un neccessary since we don't use prev_iter
        # self.x = self.x.cpu()
        # self.L = [tensor.detach() for tensor in self.L]
        # self.x_gen_logits = [tensor[0].detach().cpu() for tensor in self.x_gen_logits]
        # Return self after detaching everything
        return self

if __name__ == '__main__':
    a_prev = [0, 2, 3, 4]
    a_prev = torch.zeros((len(a_prev), 4)).scatter_(
        1, torch.clamp(torch.tensor(a_prev).unsqueeze(1) - 1, min=0),
        1.0 * (torch.tensor(a_prev).unsqueeze(1) > 0))
    print(a_prev)

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

    # -----------------------------------

    # g = torch.randn()

    # import random
    #
    # ng = 20
    # nxf = 20
    # form_memory =  FormMemory(1, [ng * nxf], lamda=0.9999, yita=0.5)
    # retrive_memory = RetriveMemory(1, 20, [ng*nxf], kappa=0.8)
    # # [W_tile xf_dimension, p_dimension]
    # W_tile = torch.tensor(np.kron(np.ones((1, nxf)),
    #                                    np.eye(ng)), dtype=torch.float)
    #
    # from scipy.stats import ortho_group
    # A = np.float32(ortho_group.rvs(dim=ng))
    # print(np.max(A), np.min(A))
    # A = torch.FloatTensor(A)
    # B = np.float32(ortho_group.rvs(dim=nxf))
    # B = torch.FloatTensor(B)
    #
    # T = 10
    # # g = [torch.randn(1, ng) for _ in range(T)]
    # g = [A[i].reshape(1, ng)
    #       for i in range(T)]
    # # xf = [torch.randn(1, nxf) for _ in range(T)]
    # xf = [B[i].reshape(1, nxf)
    #      for i in range(T)]
    #
    # M = torch.zeros(1, ng*nxf, ng*nxf)
    # for iter in range(100):
    #     for t in range(T):
    #         p = [(torch.t(xf[t]) * g[t]).reshape(1, ng*nxf)]
    #         p_ = retrive_memory([torch.matmul(g[t], W_tile)], M)
    #         M = form_memory(p, p_, M)
    #         print(torch.sum(torch.abs(p[0] - p_[0])))
    #         # print(p[0], p_[0])
    #     print('-------------')


# todo: vide into multi-modules
# todo: the problem of path integration or Generator/Inference/Tem/Model
# todo inferEntor has been temporally revised