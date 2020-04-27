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
    data=[]
    check_state=[]
    cost=[]
    for T in range(15,32):
        n_games = 5000
        num_game = 1
        start_time=time.time()
        print('------------------Max step = %s-----------------------'%T)
        while num_game <= n_games:
            if num_game%1000==0:
                print('Progress log: %s games using %s'%(num_game,time.time()-start_time))
            temp,_ = agent.collect_data(env, T)
            data.append(temp)
            temp, _ = agent.collect_data(play_env, T,True)
            reshaped_data=(list(np.array(temp[0])[:,:,:,0]),temp[1])
            data.append(reshaped_data)
            num_game = num_game + 1

    np.random.shuffle(data)

    return data

def train_load2():
    data=[]
    check_state=[]
    cost=[]
    for T in range(31,32):
        n_games = 10000
        num_game = 1
        while num_game <= n_games:
            if num_game%1000==0:
                print('Progress log: %s games'%num_game)
            temp,_ = agent.collect_data(env, T)
            length=31-len(temp[0])
            for state in temp[0]:
                fla_state=state.tolist()
                if fla_state not in check_state:
                    check_state.append(fla_state)
                    cost.append([length])
                    data.append(state)
                else:
                    idx=check_state.index(fla_state)
                    cost[idx].append(length)

            temp, _ = agent.collect_data(play_env, T,True)
            reshaped_data=list(np.array(temp[0])[:,:,:,0])
            length = 31 - len(reshaped_data)
            for state in reshaped_data:
                fla_state=state.tolist()
                if fla_state not in check_state:
                    check_state.append(fla_state)
                    cost.append([length])
                    data.append(state)
                else:
                    idx=check_state.index(fla_state)
                    cost[idx].append(length)
            # if len(temp)<31:
            #     dis=31-len(temp[0])
            #     new_cost = []
            #     for idx, c in enumerate(temp[1]):
            #         if idx > (len(temp[1]) - dis):
            #             new_cost.append(c + idx - (len(temp[1]) - dis))
            #         else:
            #             new_cost.append(c)
            #     data.append((temp[0], new_cost))
            # else:
            #data.append(reshaped_data)
            num_game = num_game + 1

    new_cost=[np.mean(c) for c in cost]

    # data_com=[]
    # data_com_count=[]
    # for data_b in data_bck:
    #     if len(data_b[0])<20:
    #         continue
    #     temp_list=data_b[0][19].tolist()
    #     if temp_list not in data_com:
    #         data_com.append(temp_list)
    #         data_com_count.append(1)
        # else:
        #     data_com_count[data_com.index(temp_list)]+=1

    # for data_f in data_for:
    #     if len(data_f[0])<13:
    #         continue
    #     temp_list=data_f[0][12].tolist()
    #     if temp_list in data_com:
        #     data_com.append(temp_list)
        #     data_com_count.append(1)
        # else:
        #     data_com_count[data_com.index(temp_list)]+=1

    return ([data],new_cost)

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

def test_DCA_eval(num_exp):
    print('---------------------------------------START TRAINING FOR EXP %s---------------------------------------' % num_exp)
    filepath='C:/Users/anvyl/Desktop/Peg_solitaire_DCA/data/'+('exp_%s_15_32_bandf_10000.pkl'%num_exp)
    print('Start data generataion')
    start_time=time.time()
    data = train_load()
    print('finish data genertaion at %s' %(time.time()-start_time))
    pickle.dump(data, open(filepath, "wb"), protocol=-1)
    states = [s[0] for s in data]
    cost = [s[1] for s in data]
    data = []
    costs = [c for cos in cost for c in cos]
    #states,costs=data
    costs_exp = np.expand_dims(costs, 1)
    states_flatten = DCAnet.state_to_nnet_input(states)
    device = DCAnet.get_device()[0]
    nnet.to(device)
    states_data = (states_flatten, costs_exp)
    batch_size=5000
    print('Start training for %s iterations' %700)
    DCAnet.tarin_nnet(nnet, states_data, device, False,batch_size, 1000,
                      train_itr=0)
    torch.save(nnet.state_dict(), "%s_model_state_dict_noupdate.pt"%num_exp )

    states=[]
    states_flatten=[]
    costs=[]
    costs_exp=[]
    states_data=()


    print('Start evaluations')
    heu = DCAnet.get_heuristic_fn(nnet, device)
    play_env = Env()
    # bck_env = DCAEnv()
    # arg=(play_env,bck_env,heu)
    # num,found=agent.play2(arg)
    # print(num,found)
    results = agent.evaluate(play_env, heu,10, 10)
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
    print('---------------------------------------FINISH TRAINING FOR EEXP %s---------------------------------------\n'%num_exp)

    #

    # rand_agent=RandomAgent()
    # play_env = Env()
    # res=rand_agent.evaluate(play_env,500,10)
    # mean_res = {}
    # for key, val in res.items():
    #     mean_res[key] = np.mean(val)
    #     if key=='pegs_left':
    #         plt.hist(val, bins='auto')
    #         plt.xlabel('num of pegs left')
    #         plt.ylabel('num of games')
    #         plt.title('Random Agent')
    #         plt.show()
    #         min_peg=min(val)
    #         count = len([i for i in val if i < mean_res['pegs_left']])
    # print('result of RandomAgent:')
    # print(mean_res)
    # print('minimum number of pegs left: ', min_peg, ' count (less than mean): ', count)

def multi_run():
    for i in range(10):
        test_DCA_eval(i)
multi_run()

#train_load()
