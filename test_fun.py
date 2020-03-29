from __future__ import print_function
import numpy as np
import tensorflow as tf
from time import time, sleep
from tqdm import tqdm
import os
import pickle
import logging
import warnings
from copy import deepcopy
from multiprocessing.dummy import Pool as ThreadPool
from random import shuffle
import torch
import torch.nn as nn

from env.DCAenv import DCAEnv
from env.env import Env
from agent import ActorCriticAgent, RandomAgent,DCAAgent
from util import read_config, flush_or_create
from buffer import Buffer
from network import DCAnet
import sys
import time

config = read_config("config.yaml")
agent_config = config['Agent']
network_config = agent_config['Network']

agent = DCAAgent(agent_config,network_config)
env=DCAEnv()
nnet: nn.Module = DCAnet.get_nnet_model()

def test_load():
    data = []
    for T in range(1, 32):
        n_games = 10
        num_game = 1
        while num_game <= n_games:
            temp,_ = agent.collect_data(env, T)
            data.append(temp)
            num_game = num_game + 1

    return data

def test_convert_nn_input():
    data=test_load()
    states=[s[0] for s in data]
    flatten=DCAnet.state_to_nnet_input(states)
    return flatten

def test_convert_torch_input():
    data = test_load()
    states = [s[0] for s in data]
    flatten = DCAnet.state_to_nnet_input(states)
    device=DCAnet.get_device()[0]
    torch=DCAnet.state_to_torch_input(flatten,device)
    print(torch)

def test_train():
    data = test_load()
    states = [s[0] for s in data]
    cost = [s[1] for s in data]
    costs = [c for cos in cost for c in cos]
    costs_exp= np.expand_dims(costs,1)
    states_flatten = DCAnet.state_to_nnet_input(states)
    device = DCAnet.get_device()[0]
    nnet.to(device)
    states_data=(states_flatten,costs_exp)
    DCAnet.tarin_nnet(nnet, states_data, device, False, 400, 300,
                          train_itr=0)

test_train()