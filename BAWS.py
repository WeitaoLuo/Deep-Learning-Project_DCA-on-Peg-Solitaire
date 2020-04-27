import numpy as np
from env.env import Env
from env.DCAenv import DCAEnv
from network import DCAnet
from util import read_config
from agent import DCAAgent
import time
class Node():
    def __init__(self,cur_state,parent,ctg,cost):
        self.cur_state=cur_state
        self.parent=parent
        self.ctg=ctg
        self.cost=cost

    def get_children_state(self):
        new_states = []
        env=Env(mid=True,mid_state=self.cur_state)
        actions = np.argwhere(env.feasible_actions)
        for action in actions:
            new_state = env.get_new_state(action)
            new_states.append([new_state[:, :, 0]])
        return new_states


def generate_goal_states():
    config = read_config("config.yaml")
    agent_config = config['Agent']
    network_config = agent_config['Network']
    agent = DCAAgent(agent_config, network_config)
    env = DCAEnv()

    new_goal_states=[]
    new_goal_path=[]

    start_time=time.time()
    print('Start generating new goal states (5 steps away)')
    for T in range(9, 10):
        n_games = 10000
        num_game = 1
        while num_game <= n_games:
            temp, _ = agent.collect_data(env, T)
            new_goal_state=temp[0][-1].tolist()
            if new_goal_state not in new_goal_states:
                new_goal_states.append(new_goal_state)
                new_goal_path.append(temp[0])
            num_game = num_game + 1
    print('Finish generating new goal states (5 steps away) @ %s'%(time.time()-start_time))
    print('# of new goal states is %s'%(len(new_goal_states)))
    return new_goal_states,new_goal_path

def Astart(heuristic,lambda_):
    open=[]
    close=[]
    ctg_list = []
    new_goal_states,new_goal_path=generate_goal_states()
    play_env=Env()
    init_states_flatten = DCAnet.state_to_nnet_input([play_env.state[:,:,0]])
    open.append(Node(play_env.state[:,:,0],None,heuristic([init_states_flatten]),0))
    cur_expand_node = open.pop(0)

    while cur_expand_node.cur_state.tolist() not in new_goal_states:
        close.append(cur_expand_node)
        child_states=cur_expand_node.get_children_state()
        if child_states !=[]:
            child_states_flatten=DCAnet.state_to_nnet_input(child_states)
            child_ctg=heuristic(child_states_flatten)
            for child,ctg in zip(child_states,child_ctg):
                if (np.count_nonzero(child[0] == 1))>=10:
                    child_node=Node(child[0],cur_expand_node,ctg,1+cur_expand_node.cost)
                    open.append(child_node)
                    ctg_list.append(ctg+(1+cur_expand_node.cost)*lambda_)

        min_ctg=np.argmin(ctg_list)
        ctg_list.pop(min_ctg)
        cur_expand_node=open.pop(min_ctg)
        play_env = Env(mid=True, mid_state=cur_expand_node.cur_state)
        if len(close)%10000==0:
            print('Progress log:')
            print('Length of close = %s' %(len(close)))
        # if cur_expand_node.get_children_state()==[]:
        #     print('DEAD END ENCOUNTER, Num of pegs left is %s'%play_env.n_pegs)
        #     print(cur_expand_node.cur_state)
        #     print('\n')

    previous_path_idx=new_goal_states.index(cur_expand_node.cur_state.tolist())
    previous_path=new_goal_path[previous_path_idx]
    path=[]
    path.append(cur_expand_node.cur_state)
    while cur_expand_node.parent!=None:
        path.append(cur_expand_node.parent.cur_state)
        cur_expand_node=cur_expand_node.parent

    return previous_path,path,len(close)+len(open)

def main():
    model_file='C:/Users/anvyl/Desktop/Peg_solitaire_DCA/best_performance/1_model_state_dict_noupdate_3peg.pt'
    nnet = DCAnet.load_nnet(model_file, DCAnet.get_nnet_model())
    device = DCAnet.get_device()[0]
    nnet.to(device)
    heu = DCAnet.get_heuristic_fn(nnet, device)
    start_time=time.time()
    previous_path,path,node_expanded=Astart(heu,0.001)
    print('Solution found using %s' %(time.time()-start_time))
    print(path)
    print(previous_path.reverse())
    print(node_expanded)



main()
