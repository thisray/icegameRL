# -*- coding: utf-8 -*-
import sys
import numpy as np
import gym
import gym_icegame

from constants import ACTION_SIZE

class GameState(object):
    def __init__(self, rand_seed, display=False):
        self.env = gym.make('IceGameEnv-v0')
        ## start here?
        # init_site = 100
        # self.env.start(init_site)
        self.reset()
        
    def reset(self):
        self.s_t = self.env.reset()
        #_, _, self.s_t = self._process_frame(7)
        self.reward = 0
        self.terminal = False
        self.action_dict = {}
        #self.s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
        
    def process(self, action):
        # convert original 18 action index to minimal action set index
        # r, t, self.s_t1, action_dict = self._process_frame(action)

        next_state, reward, done, action_dict = self.env.step(action) 

        self.reward = reward
        self.terminal = done
        self.s_t1 = next_state
        self.action_dict = action_dict
        #self.s_t1 = np.append(self.s_t[:,:,1:], x_t1, axis = 2)    

    def update(self):
        self.s_t = self.s_t1


    # def _process_frame(self, action):
    #     timeout = self.env.timeout()
    #     #x_t, reward, done, info = self.env.step(action)
    #     x_t, reward, done, action_dict = self.env.step_auto(action)
    #     terminal = timeout or done
    #     # x_t *= (1.0/255.0)
    #     return reward, terminal, x_t, action_dict
