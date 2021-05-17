# Import Absolutes deps
from os import stat
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from rlbench.backend.observation import Observation
from typing import List
import numpy as np
from torch.utils.data.dataset import Dataset
import sys

from .DDPG.ddpg import DDPG,DDPGArgs

import torch
from typing import List
import numpy as np
from rlbench.backend.observation import Observation

class LearningAgent():
    """
    General Purpose class to abstract the functionality of the network from the agent.

    Use this as a base class to create differnt Learning Based agenets that can work and be trained on 
    
    different Deep Learning Algorithms. 
    """

    def __init__(self,collect_gradients=False):
        self.learning_rate = None
        self.neural_network = None
        self.optimizer = None
        self.loss_function = None
        self.training_data = None
        self.input_state = None
        self.output_action = None
        self.total_train_size = None # This is to mark the size of the training data for the agent. 
        self.collect_gradients = collect_gradients
        self.gradients = {
            'max':[],
            'avg':[],
            'layer':[]
        }
        self.print_every = 40

    def save_model(self,file_path):
        """
        This will be used to save the model for which ever type of agent(TF/Torch)
        """
        raise NotImplementedError()

    def load_model(self,file_path):
        """
        This will be used to load the model from file.
        """
        raise NotImplementedError()
    

    def load_model_from_object(self,state_dict):
        """
        This will be used to load the model from a dictionary.
        """
        raise NotImplementedError()


class RLAgent(LearningAgent):
    def __init__(self,warmup=500, **kwargs):
        self.warmup = warmup
        self.is_training = False
        super(RLAgent,self).__init__(**kwargs)
    
    def observe(self,state_t1:List,action_t,reward_t:int,done:bool):
        """
        This is for managing replay storing. 
        Will be called after agent takes step and reward is recorded from the env. 
        This will get state: s_t+1,a_t,r_t
        """
        raise NotImplementedError()
    
    def update(self):
        """
        This will be used by the RL agents to actually Update the Policy. 
        This will let pytorch DO GD basis rewards when running the network. 
        """
        raise NotImplementedError()
    
    def act(self,state:List,**kwargs):
        """
        This will be used by the RL agents to act on state `s_t`
        This method will be used in coherance with `observe` which will get `s_t+1` as input
        This will let pytorch hold gradients when running the network. 
        """
        raise NotImplementedError()

    def reset(self,state:List,**kwargs):
        """
        This will reset the state on termination of an episode. 
        This will ensure that agent captures termination conditions of completion
        """
        raise NotImplementedError()

    
class TorchAgent(LearningAgent):

    def __init__(self,**kwargs):
        super(TorchAgent,self).__init__(**kwargs)

    def save_model(self,file_path):
        if not self.neural_network:
            return
        self.neural_network.to('cpu')
        torch.save(self.neural_network.state_dict(), file_path)

    def load_model(self,file_path):
        if not self.neural_network:
            return
        # $ this will load a model from file path.
        self.neural_network.load_state_dict(torch.load(file_path))
    

    def load_model_from_object(self,state_dict):
        if not self.neural_network:
            return 
        self.neural_network.load_state_dict(state_dict)
    
    # Expects Named Params from Torch NN Module. 
    def set_gradients(self,named_parameters):
        avg_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                avg_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        self.gradients['max'].append(max_grads)
        self.gradients['avg'].append(avg_grads)
        self.gradients['layer'].append(layers)


class TorchRLAgent(TorchAgent,RLAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



class DDPGAgent():
    # num_input_states (7+2) : 6 position + 1 gripper state + (x,y) position
    # prediction : JV + gripper state
    def __init__(self,num_input_states=9,num_output_states=7):
        pose_args = DDPGArgs()
        pose_args.bounding_min = -1
        pose_args.bounding_max= 1
        self.postion_predictor = DDPG(num_input_states,num_output_states,args=pose_args)
    
    def update(self):
        self.postion_predictor.update_policy()
    
    def observe(self,s_t,a_t, r_t, done):
        self.postion_predictor.observe(s_t,a_t,r_t,done)    
    
    def random_action(self):
        return self.postion_predictor.random_action() 

    def select_action(self,s_t):
        return self.postion_predictor.select_action(s_t)

    def reset(self):
        self.postion_predictor.reset()
        
 
class RobotAgent(TorchRLAgent):
    """
    RobotAgent
    -----------------------
    Algo Of Choice : https://spinningup.openai.com/en/latest/algorithms/ddpg.html

    Why DDPG : 
    ----------    
    So as its a continous actionspace one can try and use DDPG 
    https://math.stackexchange.com/questions/3179912/policy-gradient-reinforcement-learning-for-continuous-state-and-action-space
    https://ai.stackexchange.com/questions/4085/how-can-policy-gradients-be-applied-in-the-case-of-multiple-continuous-actions

    Inputstate 
    ----------
    num_input_states (7+2) : 6 position + 1 gripper state + (x,y) position

    prediction 
    ------------
    JV + gripper state
    """
    def __init__(self,learning_rate = 0.001,batch_size=64,collect_gradients=False,warmup=10):
        super(RobotAgent,self).__init__(collect_gradients=collect_gradients,warmup=warmup)
        self.learning_rate = learning_rate
        # action should contain 1 extra value for gripper open close state
        self.neural_network = DDPGAgent(num_input_states=9,num_output_states=7) # 1 DDPG Setup with Different Predictors. 
        self.agent_name ="DDPG_AGENT"
        self.input_state = 'joint_positions'
        self.output_action = 'joint_velocities'
        self.data_loader = None
        self.dataset = None
        self.batch_size =batch_size
        self.print_every = 40
   
    
    def _get_state_vector(self,state_dict:dict):
        return np.concatenate(
            (np.array(state_dict['joint_robot_position']),\
                np.array(state_dict['joint_gripper']),\
                    np.array(state_dict['final_postion']))
        )
        

    def observe(self,state_t:dict,action_t,reward_t:int,done:bool):
        """
        State s_t1 can be None because of errors thrown by policy. 
        """
        
        self.neural_network.observe(
            self._get_state_vector(state_t),action_t,reward_t,done
        )
    
    def update(self):
        self.neural_network.update()

    def reset(self):
        self.neural_network.reset()
        

    def act(self,state_t:dict):
       return self.neural_network.select_action(self._get_state_vector(state_t))