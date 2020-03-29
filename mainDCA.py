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

warnings.filterwarnings("ignore")

def data_load(agent,env,batch_size):
    data = []
    for T in range(1, 32):
        n_games = batch_size
        num_game = 1
        while num_game <= n_games:
            temp,_ = agent.collect_data(env, T)
            data.append(temp)
            num_game = num_game + 1

    return data

def main():
	# arguments
	config = read_config("config.yaml")
	agent_config = config['Agent']
	network_config = agent_config['Network']

	agent = DCAAgent(agent_config, network_config)
	env = DCAEnv()
	nnet: nn.Module = DCAnet.get_nnet_model()

	# get device
	on_gpu: bool
	device: torch.device
	device, devices, on_gpu = DCAnet.get_device()
	print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

	save_dir='C:/Users/anvyl/Desktop/RL-solitaire-master/saved_models'
	nnet_name='DCA'
	model_dir = "%s/%s/" % (save_dir, nnet_name)

	# training
	itr = 0
	num_updates: int = 0

	batch_size =300
	update_num=0
	max_updates=10
	epochs_per_update=1

	while update_num < max_updates:
		data = []
		states = []
		cost = []
		costs =[]

		start_time=time.time()
		print('Starting data generation')

		data = data_load(agent,env,100)
		print('Finish data generation at %s' % (time.time() - start_time))
		states = [s[0] for s in data]
		cost = [s[1] for s in data]
		costs = [c for cos in cost for c in cos]
		costs_exp = np.expand_dims(costs, 1)
		states_flatten = DCAnet.state_to_nnet_input(states)

		states_data = (states_flatten, costs_exp)

		# num_updates += 1

		# load nnet
		start_time = time.time()
		model_save_loc = "%s/%s" % (model_dir, update_num)
		model_file = "%s/model_state_dict.pt" % model_save_loc
		if os.path.isfile(model_file):
			nnet = DCAnet.load_nnet(model_file, DCAnet.get_nnet_model())
		else:
			nnet: nn.Module = DCAnet.get_nnet_model()

		nnet.to(device)
		# if on_gpu and not args_dict['single_gpu_training']:
		# 	nnet = nn.DataParallel(nnet)
		print("Load nnet time: %s" % (time.time() - start_time))

		# train nnet
		num_train_itrs: int = epochs_per_update * np.ceil(len(states_data[0]) / batch_size)
		#num_train_itrs=10
		print("Training model for update number %i for %i iterations" % (update_num, num_train_itrs))
		DCAnet.tarin_nnet(nnet, states_data, device, on_gpu, batch_size, num_train_itrs,
							  train_itr=itr)
		itr += num_train_itrs

		if not os.path.exists(model_save_loc):
			os.makedirs(model_save_loc)

		# save nnet
		torch.save(nnet.state_dict(), "%s/model_state_dict.pt" % model_save_loc)

		# test
		#
		# if num_updates % args_dict['testing_freq'] == 0:
		# heuristic_fn = nnet_utils.get_heuristic_fn(nnet, device, env)
		# 	gbfs_test(args_dict['val_dir'], env, heuristic_fn, max_solve_steps=args_dict['testing_solve_steps'])
		# clear cuda memory
		del nnet
		torch.cuda.empty_cache()

		update_num = update_num + 1

	print("Done")

if __name__ == '__main__':
	main()