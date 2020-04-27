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
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def data_load(agent,env,num_games,forward,steps):
	data = []
	if forward:
		scramble_range=31
	else:
		scramble_range=steps
	for T in range(scramble_range, scramble_range+1):
		n_games = num_games
		num_game = 1
		while num_game <= n_games:
			if forward==False:
				temp,_ = agent.collect_data(env, T)
				data.append(temp)
			else:
				temp,_=agent.collect_data(env, T, True)
				end_pt=min(len(temp[0]),32-steps)
				reshaped_data = (list(np.array(temp[0])[:, :, :, 0][-end_pt:]), temp[1][-end_pt:])
				data.append(reshaped_data)
			num_game = num_game + 1
	return data

def data_load2(agent,env,batch_size,forward):
	data = []
	for T in range(1, 32):
		n_games = batch_size
		num_game = 1
		while num_game <= n_games:
			if forward==False:
				temp,_ = agent.collect_data(env, T)
				data.append(temp)
			else:
				temp,_=agent.collect_data(env, T, True)
				reshaped_data = (list(np.array(temp[0])[:, :, :, 0]), temp[1])
				data.append(reshaped_data)
			num_game = num_game + 1
	return data

def main():
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

	save_dir='C:/Users/anvyl/Desktop/Peg_solitaire_DCA/saved_models'
	nnet_name='DCA'
	model_dir = "%s/%s" % (save_dir, nnet_name)

	# training
	itr = 0

	batch_size =500
	update_num=0
	max_updates=5
	epochs_per_update=1

	while update_num < max_updates:
		print('################ START TRAINING FOR EPOCH %s ######################'% (update_num))
		data = []
		states = []
		cost = []
		costs =[]
		bck_env=DCAEnv()
		for_env=Env()
		start_time=time.time()

		model_save_loc = "%s/%s" % (model_dir, update_num)
		model_load_loc = "%s/%s" % (model_dir, update_num - 1)
		model_file = "%s/model_state_dict.pt" % model_load_loc
		if (os.path.isfile(model_file)) == True:
			nnet = DCAnet.load_nnet(model_file, DCAnet.get_nnet_model())
		else:
			nnet: nn.Module = DCAnet.get_nnet_model()
		nnet.to(device)


		print('-----------------------------Start training for backwards------------------------------------')
		for i in range(1,32):
			print("Start for %s steps scrambles"%i)
			print('Starting data generation')
			start_time = time.time()
			if i<10:
				num_sample=1000
			else:
				num_sample=1000
			data = data_load(agent,bck_env,num_sample,False,i)
			print('Finish data generation at %s' % (time.time() - start_time))
			states = [s[0] for s in data]
			cost = [s[1] for s in data]
			costs = [c for cos in cost for c in cos]
			costs_exp = np.expand_dims(costs, 1)
			states_flatten = DCAnet.state_to_nnet_input(states)

			states_data = (states_flatten, costs_exp)
			if (i>1):
				nnet = DCAnet.load_nnet("%s/model_state_dict.pt" % model_save_loc, DCAnet.get_nnet_model())
				nnet.to(device)

			# train nnet
			num_train_itrs: int = epochs_per_update * np.ceil(len(states_data[0]) / batch_size)
			#num_train_itrs=10
			DCAnet.tarin_nnet(nnet, states_data, device, on_gpu, batch_size, num_train_itrs,
							  train_itr=itr,Update=True)
			itr += num_train_itrs

			# save nnet
			if not os.path.exists(model_save_loc):
				os.makedirs(model_save_loc)
			torch.save(nnet.state_dict(), "%s/model_state_dict.pt" % model_save_loc)
			print("Finish for %s steps scrambles\n" % i)
		print('-----------------------------Finish training for backwards------------------------------------\n')

		if not os.path.exists(model_save_loc):
			os.makedirs(model_save_loc)
		torch.save(nnet.state_dict(), "%s/model_state_dict.pt" % model_save_loc)
		
		#------------------Training for forward data-------------------------------------------------
		print('-----------------------------Start training for forwards------------------------------------\n')
		for j in range(31, 0,-1):
			print("Start for %s steps scrambles" % j)
			print('Starting data generation')
			start_time=time.time()
			data = data_load(agent, for_env, 1000, True,j)
			data.reverse()
			print('Finish data generation at %s' % (time.time() - start_time))
			states = [s[0] for s in data]
			cost = [s[1] for s in data]
			costs = [c for cos in cost for c in cos]
			costs_exp = np.expand_dims(costs, 1)
			states_flatten = DCAnet.state_to_nnet_input(states)

			states_data = (states_flatten, costs_exp)


			-nnet = DCAnet.load_nnet("%s/model_state_dict.pt" % model_save_loc, DCAnet.get_nnet_model())
			nnet.to(device)

			# train nnet
			num_train_itrs: int = epochs_per_update * np.ceil(len(states_data[0]) / batch_size)
			# num_train_itrs=10
			DCAnet.tarin_nnet(nnet, states_data, device, on_gpu, batch_size, num_train_itrs,
						  train_itr=itr,Update=True,Forward=True)

			itr += num_train_itrs
			print("Finish for %s steps scrambles\n" % j)
			if not os.path.exists(model_save_loc):
				os.makedirs(model_save_loc)
			torch.save(nnet.state_dict(), "%s/model_state_dict.pt" % model_save_loc)


		print('-----------------------------Finish training for forward------------------------------------')

		# save nnet
		# if not os.path.exists(model_save_loc):
		# 	os.makedirs(model_save_loc)
		# torch.save(nnet.state_dict(), "%s/model_state_dict.pt" % model_save_loc)

		# test
		#
		play_env = Env()
		heu = DCAnet.get_heuristic_fn(nnet, device)
		results = agent.evaluate(play_env, heu, 100, 10)
		mean_res = {}
		for key, val in results.items():
			mean_res[key] = np.mean(val)
			if key == 'pegs_left':
				plt.hist(val, bins='auto')
				plt.xlabel('num of pegs left')
				plt.ylabel('num of games')
				plt.title('DCA Agent')
				plt.show()
				min_peg = min(val)
				count = len([i for i in val if i < mean_res['pegs_left']])

		print('result of prelimary DCAAgent:')
		print(mean_res)
		print('minimum number of pegs left: ', min_peg, ' count (less than mean): ', count)

		del nnet
		torch.cuda.empty_cache()


		print('################ FINISH TRAINING FOR EPOCH %s #############' % (update_num))
		update_num = update_num + 1

	print("Done")

if __name__ == '__main__':
	main()