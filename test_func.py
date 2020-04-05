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
from matplotlib import pyplot as plt

config = read_config("config.yaml")
agent_config = config['Agent']
network_config = agent_config['Network']

agent = DCAAgent(agent_config,network_config)
env=DCAEnv()
play_env=Env()
nnet: nn.Module = DCAnet.get_nnet_model()

def train_load():
    data = []
    for T in range(1, 32):
        n_games = 1000
        num_game = 1
        while num_game <= n_games:
            temp,_ = agent.collect_data(env, T)
            data.append(temp)
            num_game = num_game + 1

    return data

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
    data=train_load()
    states=[s[0] for s in data]
    flatten=DCAnet.state_to_nnet_input(states)
    return flatten

def test_convert_torch_input():
    data = train_load()
    states = [s[0] for s in data]
    flatten = DCAnet.state_to_nnet_input(states)
    device=DCAnet.get_device()[0]
    torch=DCAnet.state_to_torch_input(flatten,device)
    print(torch)

def test_train():
    data = train_load()
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

def test_load_heu():
    data = train_load()
    states = [s[0] for s in data]
    cost = [s[1] for s in data]
    costs = [c for cos in cost for c in cos]
    costs_exp = np.expand_dims(costs, 1)
    states_flatten = DCAnet.state_to_nnet_input(states)
    device = DCAnet.get_device()[0]
    nnet.to(device)
    states_data = (states_flatten, costs_exp)
    DCAnet.tarin_nnet(nnet, states_data, device, False, 400, 300,
                      train_itr=0)

    test_data=test_load()
    test_states = [s[0] for s in test_data]
    test_cost = [s[1] for s in test_data]
    test_costs = [c for cos in test_cost for c in cos]
    #costs_exp = np.expand_dims(costs, 1)
    test_states_flatten = DCAnet.state_to_nnet_input(test_states)
    heu = DCAnet.get_heuristic_fn(nnet, device)
    pred=heu(test_states_flatten)
    print(pred)


def test_naive_policy():
    data = train_load()
    states = [s[0] for s in data]
    cost = [s[1] for s in data]
    costs = [c for cos in cost for c in cos]
    costs_exp = np.expand_dims(costs, 1)
    states_flatten = DCAnet.state_to_nnet_input(states)
    device = DCAnet.get_device()[0]
    nnet.to(device)
    states_data = (states_flatten, costs_exp)
    DCAnet.tarin_nnet(nnet, states_data, device, False, 400, 300,
                      train_itr=0)

    heu = DCAnet.get_heuristic_fn(nnet, device)
    action=agent.naive_policy(play_env,heu,play_env.feasible_actions)
    play_env.step(action)
    print(play_env.state[:,:,0])

def test_DCA_eval():
    data = train_load()
    states = [s[0] for s in data]
    cost = [s[1] for s in data]
    costs = [c for cos in cost for c in cos]
    costs_exp = np.expand_dims(costs, 1)
    states_flatten = DCAnet.state_to_nnet_input(states)
    device = DCAnet.get_device()[0]
    nnet.to(device)
    states_data = (states_flatten, costs_exp)
    DCAnet.tarin_nnet(nnet, states_data, device, False, 1000, 300,
                      train_itr=0)

    heu = DCAnet.get_heuristic_fn(nnet, device)
    results = agent.evaluate(play_env, heu,500, 10)
    mean_res={}
    for key,val in results.items():
        mean_res[key]=np.mean(val)
        if key=='pegs_left':
            plt.hist(val,bins='auto')
            plt.xlabel('num of pegs left')
            plt.ylabel('num of games')
            plt.title('DCA Agent')
            plt.show()
            min_peg=min(val)
            count = len([i for i in val if i < mean_res['pegs_left']])

    print('result of prelimary DCAAgent:')
    print(mean_res)
    print('minimum number of pegs left: ', min_peg, ' count (less than mean): ', count)


    rand_agent=RandomAgent()
    res=rand_agent.evaluate(play_env,500,10)
    mean_res = {}
    for key, val in res.items():
        mean_res[key] = np.mean(val)
        if key=='pegs_left':
            plt.hist(val, bins='auto')
            plt.xlabel('num of pegs left')
            plt.ylabel('num of games')
            plt.title('Random Agent')
            plt.show()
            min_peg=min(val)
            count = len([i for i in val if i < mean_res['pegs_left']])
    print('result of RandomAgent:')
    print(mean_res)
    print('minimum number of pegs left: ', min_peg, ' count (less than mean): ', count)

test_DCA_eval()