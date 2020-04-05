from typing import List, Tuple
import numpy as np
import os
import torch
from torch import nn

from network.pytorch_models import ResnetModel
from collections import OrderedDict
import re
from random import choice
from torch import Tensor

import torch.optim as optim
from torch.optim.optimizer import Optimizer

import time


def get_nnet_model() -> nn.Module:
    state_dim: int = 7*7 # Area of the board
    nnet = ResnetModel(state_dim, 6, 5000, 1000, 4, 1)

    return nnet


def state_to_nnet_input( states: List) -> List[List[np.ndarray]]:
    states_np = np.stack([s for state in states for s in state], axis=0)

    representation_np: np.ndarray = states_np
    # representation_np: np.ndarray = representation_np.astype(self.dtype)

    representation: List[List[np.ndarray]] = [[x] for x in representation_np]

    return representation


def state_to_torch_input(states_nnet:List[List],device)->List[Tensor]:
    states_nnet_tensors=[]
    for tensor_ind in range(len(states_nnet[0])):
        tensor_np = np.stack(temp[tensor_ind] for temp in states_nnet)
        tensor = torch.tensor(tensor_np,device=device)

        states_nnet_tensors.append(tensor)

    return states_nnet_tensors

def generate_batch(data:Tuple[List,np.ndarray],batch_size:int) ->List[Tuple[np.ndarray,np.ndarray]]:
    rand_ind=np.random.choice(len(data[0]),len(data[0]),replace=False)
    data_input:List=[data[0][i] for i in rand_ind]
    data_output=data[1][rand_ind].astype(np.float32)

    data_batches=[]
    start_ind=0
    while (start_ind+batch_size)<=len(data[0]):
        end_ind=start_ind+batch_size
        input_batch=data_input[start_ind:end_ind]
        output_batch=data_output[start_ind:end_ind]
        data_batches.append((input_batch,output_batch))
        start_ind=end_ind
    return data_batches

# def update_value(data:Tuple[List,np.ndarray],)

def tarin_nnet(nnet:nn.Module,data:Tuple[List,np.ndarray],device:torch.device,on_gpu:bool,batch_szie:int,
               num_itrs:int,train_itr:int=0,display:bool=True):
    display_itrs=10
    loss=nn.MSELoss()
    optimizer:Optimizer=optim.Adam(nnet.parameters(),lr=0.001)

    start_time=time.time()
    batches:List[Tuple[np.ndarray,np.ndarray]]=generate_batch(data,batch_szie)

    nnet.train()
    max_itrs: int = train_itr + num_itrs

    while train_itr<max_itrs:
        optimizer.zero_grad()

        x,y=choice(batches)
        y=y.astype(np.float32)

        x_tensor:List[Tensor]=state_to_torch_input(x,device)
        y_tensor:Tensor=torch.tensor(y,device=device)

        nnet_output:Tensor =nnet(*x_tensor)

        ###Not very clear
        nnet_predicted_y=nnet_output[:,0]
        true_y=y_tensor[:,0]

        nnloss=loss(nnet_predicted_y,true_y)

        nnloss.backward()

        optimizer.step()

        if (train_itr%display_itrs==0) and display:
            print("Itr: %i, loss: %.2f, targ_ctg: %.2f, nnet_ctg: %.2f, "
                  "Time: %.2f" % (
                      train_itr, nnloss.item(), true_y.mean().item(), nnet_predicted_y.mean().item(),
                      time.time() - start_time))

            start_time = time.time()

        train_itr =train_itr+1

def get_device() -> Tuple[torch.device, List[int], bool]:
    device: torch.device = torch.device("cpu")
    devices: List[int] = []
    on_gpu: bool = False
    if ('CUDA_VISIBLE_DEVICES' in os.environ) and torch.cuda.is_available():
        device = torch.device("cuda:%i" % 0)
        devices = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]
        on_gpu = True

    return device, devices, on_gpu

# loading nnet
def load_nnet(model_file: str, nnet: nn.Module, device: torch.device = None) -> nn.Module:
    # get state dict
    if device is None:
        state_dict = torch.load(model_file)
    else:
        state_dict = torch.load(model_file, map_location=device)

    # remove module prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = re.sub('^module\.', '', k)
        new_state_dict[k] = v

    # set state dict
    nnet.load_state_dict(new_state_dict)

    nnet.eval()

    return nnet

def get_heuristic_fn(nnet: nn.Module, device: torch.device, clip_zero: bool = False,
                     batch_size: int = 100):
    nnet.eval()

    def heuristic_fn(states: List) -> np.ndarray:
        cost_to_go: np.ndarray = np.zeros(0)

        num_states: int = len(states)
        start_idx: int = 0
        while start_idx < num_states:
            # get batch
            end_idx: int = min(start_idx + batch_size, num_states)
            states_batch: List = states[start_idx:end_idx]

            # convert to nnet input
            states_nnet_batch: List[List] =state_to_nnet_input(states_batch)

            # get nnet output
            states_nnet_batch_tensors =state_to_torch_input(states_nnet_batch, device)
            cost_to_go_batch: np.ndarray = nnet(*states_nnet_batch_tensors).cpu().data.numpy()

            cost_to_go: np.ndarray = np.concatenate((cost_to_go, cost_to_go_batch[:, 0]), axis=0)

            start_idx: int = end_idx

        assert (cost_to_go.shape[0] == num_states)

        if clip_zero:
            cost_to_go = np.maximum(cost_to_go, 0.0)

        return cost_to_go

    return heuristic_fn



