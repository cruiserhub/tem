import numpy as np
from itertools import count
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn import init

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML
import string
# import gensim

GOAL_VALUE = 1
EDGE_VALUE = -1
HOLE_VALUE = -1
VISIBLE_RADIUS = 1
GRID_SIZE = 8
NUM_HOLES = 4

GOAL_VALUE = 1
EDGE_VALUE = -1
HOLE_VALUE = -1
VISIBLE_RADIUS = 1
GRID_SIZE = 8
NUM_HOLES = 4


# enviroment and behaviour, without moving out of wall
class Grid():
    def __init__(self, n_holes=0, grid_size=GRID_SIZE, random_seed=0, set_reward=0):
        random.seed(random_seed)
        # check type of grid,  attention type is not a string
        if type(grid_size) == int:
            self.grid_size_y = self.grid_size_x = grid_size
        elif type(grid_size) == tuple:
            y, x = grid_size
            self.grid_size_y = y
            self.grid_size_x = x
        self.n_holes = n_holes
        #  Define the surronding using padding
        padded_size_y = self.grid_size_y + 4 * VISIBLE_RADIUS
        padded_size_x = self.grid_size_x + 4 * VISIBLE_RADIUS
        #  intialize grid with zeros, attention y and x order
        self.grid = np.zeros((padded_size_y, padded_size_x))  # Padding for edges
        #  intialize border with predefined negative values
        self.grid[0:2 * VISIBLE_RADIUS, :] = EDGE_VALUE
        self.grid[-2 * VISIBLE_RADIUS:, :] = EDGE_VALUE
        self.grid[:, 0:2 * VISIBLE_RADIUS] = EDGE_VALUE
        self.grid[:, -2 * VISIBLE_RADIUS:] = EDGE_VALUE

        # Randomly placed plants and intialize them with random values
        for i in range(self.n_holes):
            # uniform random distribution
            ry = random.randint(0, self.grid_size_y) + 2 * VISIBLE_RADIUS
            rx = random.randint(0, self.grid_size_x) + 2 * VISIBLE_RADIUS
            self.grid[ry, rx] = HOLE_VALUE
        # Goal set
        if set_reward == 0:
            gy = random.randint(0, self.grid_size_y) + 2 * VISIBLE_RADIUS
            gx = random.randint(0, self.grid_size_x) + 2 * VISIBLE_RADIUS
            while self.grid[gy, gx] == HOLE_VALUE:
                gy = random.randint(0, self.grid_size_y) + 2 * VISIBLE_RADIUS
                gx = random.randint(0, self.grid_size_x) + 2 * VISIBLE_RADIUS
            self.grid[gy, gx] = GOAL_VALUE
        else:
            for pos_reward in set_reward:
                radius = self.grid_size_y // 31
                self.grid[pos_reward[0] - radius: pos_reward[0] + 1 + radius,
                pos_reward[1] - radius: pos_reward[1] + 1 + radius] = GOAL_VALUE

    def visible1(self, pos):
        # observable area is the squre around the agent, so 3x3 region , problem is when the agent is going to the
        # edge and corner of the grid
        y, x = pos
        y_relative = y * 19. / (self.grid_size_y + 4)
        x_relative = x * 19. / (self.grid_size_x + 4)
        visible = self.grid[y - VISIBLE_RADIUS:y + VISIBLE_RADIUS + 1, x - VISIBLE_RADIUS:x + VISIBLE_RADIUS + 1]
        if np.sum(visible) != 0 and (x == 2 or x == self.grid_size_x + 1):
            visible = np.multiply(visible, y_relative * np.ones((3, 3)))
        elif np.sum(visible) != 0 and (y == 2 or y == self.grid_size_y + 1):
            visible = np.multiply(visible, x_relative * np.ones((3, 3)))
        return visible


def string_vectorizer(string, alphabet=string.ascii_lowercase):
    vector = [[0 if char != letter else 1 for char in alphabet]
              for letter in string]
    return vector


class Envs(Grid):
    def __init__(self, grid_size=15, type='default'):
        Grid.__init__(self, grid_size=grid_size)
        if type == 'default':
            self.objects = np.zeros_like(self.grid)
            for i in range(5):
                for j in range(5):
                    self.objects[i * 3 + 2, j * 3 + 2] = 1
        elif type == 'onehot':
            self.objects = np.zeros((self.grid_size_y + 2 * VISIBLE_RADIUS, self.grid_size_x + 2 * VISIBLE_RADIUS, 40))
            for i in range(self.grid_size_y + 2 * VISIBLE_RADIUS):
                for j in range(self.grid_size_x + 2 * VISIBLE_RADIUS):
                    self.objects[i, j] = np.eye(40)[j + i]
        elif type == 'objects':
            nw = 26
            self.objects_type = ['apple', 'orange', 'pear', 'banana', 'grape', 'dog', 'cat']
            self.objects_set = {w: i + 1 for i, w in enumerate(set(self.objects_type))}
            objects, pos_vect = [], []
            for i in range(100):
                objects.append(self.objects_type[np.random.randint(len(self.objects_type))])
                pos_vect.append((np.random.randint(2, self.grid_size_y + 2 * VISIBLE_RADIUS),
                                 np.random.randint(2, self.grid_size_x + 2 * VISIBLE_RADIUS)))
            self.sensory = np.zeros((self.grid_size_y + 2 * VISIBLE_RADIUS, self.grid_size_x + 2 * VISIBLE_RADIUS, nw))
            self.objects = np.zeros((self.grid_size_y + 2 * VISIBLE_RADIUS, self.grid_size_x + 2 * VISIBLE_RADIUS, 1))
            for i, pos in enumerate(pos_vect):
                ob_vect = self.encode_nhot(objects[i])
                self.sensory[pos[0] - 1: pos[0] + 2, pos[1] - 1: pos[1] + 2] = ob_vect
                self.objects[pos[0] - 1: pos[0] + 2, pos[1] - 1: pos[1] + 2] = self.objects_set[objects[i]]

        # if type == 'plus':
        #     self.objects =

    def encode_nhot(self, word):
        vect = np.sum(string_vectorizer(word), axis=0)
        return vect

    def decode_nhot(self, word_code):
        score = [np.corrcoef(word_code, self.encode_nhot(obj))[0][1] for obj in self.objects_type]
        return score

    def encode_w2v(self):
        w2v = models.Word2Vec()

    def visible(self, pos):
        # observable area is the squre around the agent, so 3x3 region , problem is when the agent is going to the
        # edge and corner of the grid
        y, x = pos
        visible = self.objects[y - VISIBLE_RADIUS:y + VISIBLE_RADIUS + 1, x - VISIBLE_RADIUS:x + VISIBLE_RADIUS + 1]
        return visible


class Agent(Envs):
    def __init__(self, grid_size=15, set_agent=0):
        Grid.__init__(self, grid_size=grid_size)
        if type(grid_size) == tuple:
            self.grid_size_y, self.grid_size_x = grid_size
        else:
            self.grid_size_y = self.grid_size_x = grid_size
        # position initialize
        if set_agent == 0:
            random.seed()
            self.pos = (np.random.randint(self.grid_size_y) + 2 * VISIBLE_RADIUS,
                        np.random.randint(self.grid_size_x) + 2 * VISIBLE_RADIUS)
        else:
            self.pos = set_agent

    def reset(self, grid_size, set_agent=0):
        if type(grid_size) == tuple:
            self.grid_size_y, self.grid_size_x = grid_size
        else:
            self.grid_size_y = self.grid_size_x = grid_size
        # position initialize
        if set_agent == 0:
            random.seed()
            self.pos = (np.random.randint(self.grid_size_y) + 2 * VISIBLE_RADIUS,
                        np.random.randint(self.grid_size_x) + 2 * VISIBLE_RADIUS)
        else:
            self.pos = set_agent

    # implement reflecting boundary
    def wall(self, pos):
        y, x = pos
        # wall detection
        if np.sum(self.grid[pos]) < 0:
            # pos_possible = [pos for pos in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)] if
            #                 self.grid[pos] >= 0]
            pos = (np.random.randint(self.grid_size_y) + 2 * VISIBLE_RADIUS,
                   np.random.randint(self.grid_size_x) + 2 * VISIBLE_RADIUS)
            return pos
        else:
            return pos

    # moves in four direction, Implement the relfective behaviour when arriving upon the wall  ,  one way is to let the agent ran randomly when it clicks to wall, the other way is reflective boundary.
    def act(self, action):
        # Move according to action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        y, x = self.pos
        #         # wall detection,  upwall
        #         if y == 2 * VISIBLE_RADIUS and action == 0:
        #             action = [1, 2, 3][np.random.randint(3)]
        # #             action = 2 - action
        #         # downwall
        #         elif y == self.grid_size_y - 1 + 2 * VISIBLE_RADIUS and action == 2:
        #             action = [0, 1, 3][np.random.randint(3)]
        # #             action = 2 - action
        #         # left wall
        #         elif x == 2 * VISIBLE_RADIUS and action == 3:
        #             action = [0, 1, 2][np.random.randint(3)]
        # #             action = 4 - action
        #         # right wall
        #         elif x == self.grid_size_x - 1 + 2 * VISIBLE_RADIUS and action == 1:
        #             action = [0, 2, 3][np.random.randint(3)]
        # #             action = 4 - action
        # up
        if action == 0:
            y -= 1
        # right
        elif action == 1:
            x += 1
        # down
        elif action == 2:
            y += 1
        # left
        elif action == 3:
            x -= 1
        pos = (y, x)
        pos = self.wall(pos)
        self.pos = pos



