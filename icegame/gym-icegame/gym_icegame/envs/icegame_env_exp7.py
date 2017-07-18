from __future__ import division
import gym
from gym import error, spaces, utils, core

#from six import StingIO
import sys
import numpy as np
import random
from libicegame import SQIceGame, INFO

#import matplotlib.pyplot as plt

import time

rnum = np.random.randint

class IceGameEnv(core.Env):
    def __init__ (self, L, kT, J):
        self.L = L
        self.kT = kT
        self.J = J
        self.N = L**2 
        num_neighbors = 2
        num_replicas = 1
        num_mcsteps = 2000
        num_bins = 1
        num_thermalization = num_mcsteps
        tempering_period = 1

        self.mc_info = INFO(self.L, self.N, num_neighbors, num_replicas, \
                num_bins, num_mcsteps, tempering_period, num_thermalization)

        self.sim = SQIceGame(self.mc_info)
        self.sim.set_temperature (self.kT)
        self.sim.init_model()
        self.sim.mc_run(num_mcsteps)

        self.episode_terminate = False
        self.accepted_episode = False

        self.name_mapping = dict({
                                  0 :   'right',
                                  1 :   'down',
                                  2 :   'left',
                                  3 :   'up',
                                  4 :   'lower_next',
                                  5 :   'upper_next',
                                  6 :   'metropolis',
                                  7 :   'noop',
                                  })

        self.index_mapping = dict({
                                  'right': 0,
                                  'down' : 1,
                                  'left' : 2,
                                  'up' : 3,
                                  'lower_next' : 4,
                                  'upper_next' : 5,
                                  'metropolis' : 6,
                                  'noop'       : 7,
                                  })

        ### action space and state space
        self.action_space = spaces.Discrete(len(self.name_mapping))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.L, self.L, 4))
        self.reward_range = (-1, 1)

        self.ofilename = 'loop_config.log'
        self.stacked_axis = 2

        ## test by ray
        self.length_dict = {}
        self.area_dict = {}
        self.action_dict = {}
        self.move_times = 0
        self.start_point = 0

    def step(self, action): # make it auto
        terminate = False
        reward = 0.0
        obs = None
        rets = [0.0, 0.0, 0.0, 0.0]

        # ## add times here
        # if self.move_times > 1024 and action != 6:
        #     # terminate = True
        #     self.move_times = 0
        #     action = 6
        #     # return self.step(6)
        #     # self.start(rnum(self.N))

        metropolis_executed = False
        self.dict_dict(target_dict=self.action_dict, target_key=action)

        if (action == 6):
            self.sim.flip_trajectory()
            rets = self.sim.metropolis()
            metropolis_executed = True

        elif (0 <= action < 6):
            rets = self.sim.draw(action)
            self.move_times = self.move_times +1

            ## auto6 HERE
            if(rets[1]==0.0 and rets[2]==0.0) or (self.move_times > 1024):
                # return self.step(6)
                self.sim.flip_trajectory()
                rets = self.sim.metropolis()
                metropolis_executed = True

        if (metropolis_executed):
            self.move_times = 0
            
            # ## test add this
            # terminate = True
            if rets[0] > 0 and rets[3] > 0:
                ## test add this
                terminate = True
                print ('ACCEPTS!')
                self.sim.update_config()
                ## get length and update dict
                accepted_length = int(self.sim.get_accepted_length()[-1])
                accepted_area = self.caculate_area()
                # self.r_dict_update(length=accepted_length, area=accepted_area)
                self.dict_dict(target_dict=self.length_dict, target_key=accepted_length)
                self.dict_dict(target_dict=self.area_dict, target_key=accepted_area)
                ## reward
                r_length = 1.0
                r_area = 8.0
                reward = r_length * accepted_length + r_area * accepted_area
                if reward > 2:
                    self.render()
                ## record
                with open(self.ofilename, 'a') as f:
                    f.write('{}\n'.format(self.sim.get_trajectory()))
                    print ('\tSave loop configuration to file')
                ## print
                print ('\tTotal accepted number = {}, loop length = {}, loop area = {}, reward = {}'.format(self.sim.get_updated_counter(), accepted_length, accepted_area, reward))
                print ('\tAccepted loop length record = {}'.format(self.length_dict))
                print ('\tAccepted loop area record = {}'.format(self.area_dict))
                print ('\trets list = {}'.format(rets))
                self.sim.reset()    ## reset positon
            else:
                if (rets[3] == 0):     # I guess this is GameOver ?
                    reward = - 0.5 * 0.1
                else:
                    reward = - 1 * 0.1
                self.sim.reset()
                # self.start(rnum(self.N))
            # reset or update
        else:
            ## walk & flip reward
            reward = self._stepwise_weighted_returns(rets)
            ## try this
            # if reward < 0:
            #     # reward = reward * 0
            #     # terminate = True
            #     # self.start(rnum(self.N))    # try
            #     reward = reward
            # as usual

        obs = self.get_obs()
        return obs, reward, terminate, self.action_dict #rets

    def _stepwise_weighted_returns(self, rets):
        # icemove_w = 0.0
        # energy_w = -10.0
        # defect_w = -10.0
        # icemove_w = 0.001
        # energy_w = -0.1 * 0.0
        # defect_w = -0.1 * 0.0      
        # return icemove_w * rets[0] + energy_w * rets[1] + defect_w * rets[2]
        reward_0 = rets[0] * 0.05
        reward_1 = - 0.1 * rets[1]
        reward_2 = - 0.1 * rets[2]

        if rets[1] == 0.0 and rets[2] == 0.0:
            reward_0 = 2.0
        elif rets[1] == 0.0:
            reward_1 = 1.0 * 0.5
        elif rets[2] == 0.0:
            reward_2 = 1.0 * 0.0
        return reward_0 + reward_1 + reward_2



    ## use dict to show Accepted loop length
    # def r_dict_update(self, length, area):
    def dict_dict(self, target_dict, target_key):
        if target_key in target_dict:
            target_dict[target_key] = target_dict[target_key] +1
        else:
            target_dict[target_key] = 1
        # dict_dict(target_dict=self.length_dict, target_key=length)
        # dict_dict(target_dict=self.area_dict, target_key=area)

    def step_binary(self, action):
        pass
        

    # Start function used for agent learing
    def start(self, init_site):
        self.move_times = 0
        self.start_point = init_site
        init_agent_site = self.sim.start(init_site)
        assert(init_site == init_agent_site)

    ## New version: clear buffer and set new start of agent
    def reset(self):
        # self.sim.clear_buffer()
        self.start_point = rnum(self.N)
        self.sim.reset()
        self.start(self.start_point)
        self.action_dict = {}
        return self.get_obs()

    # def reset(self):
    #     self.sim.reset()
    #     return self.get_obs()

    def timeout(self):
        return self.sim.timeout()

    @property
    def agent_site(self):
        return self.sim.get_agent_site()

    @property
    def action_name_mapping(self):
        return self.name_mapping

    @property
    def name_action_mapping(self):
        return self.index_mapping

    def sample_icemove_action_index(self):
        return self.sim.icemove_index()

    def render(self, mapname ='traj', mode='ansi', close=False):
        #of = StringIO() if mode == 'ansi' else sys.stdout
        #print ('Energy: {}, Defect: {}'.format(self.sqice.cal_energy_diff(), self.sqice.cal_defect_density()))
        s = None
        if (mapname == 'traj'):
            s = self._transf2d(self.sim.get_canvas_map())
        elif (mapname == 'state'):
            s = self._transf2d(self.sim.get_state_t_map())
        screen = '\r'
        screen += '\n\t'
        screen += '+' + self.L * '---' + '+\n'
        for i in range(self.L):
            screen += '\t|'
            for j in range(self.L):
                p = (i, j)
                spin = s[p]
                if spin == -1:
                    screen += ' o '
                elif spin == +1:
                    screen += ' * '
                elif spin == 0:
                    screen += '   '
                elif spin == +2:
                    screen += ' @ '
                elif spin == -2:
                    screen += ' O '
                elif spin == 100:
                    screen += ' % '
            screen += '|\n'
        screen += '\t+' + self.L * '---' + '+\n'
        sys.stdout.write(screen)


    # ray test, return walk_path_2D_dict
    def conver1Dto2D(self, path_1D):
        path_2D_dict = {}
        for position in path_1D:
            position_x = position % 32
            position_y = int(position / 32)
            
            if position_x in path_2D_dict:
                path_2D_dict[position_x].append(position_y)
            else:
                path_2D_dict[position_x] = [position_y]
        return path_2D_dict

    # ray test
    def caculate_area(self):
        area = 0
        walk_path_1D = self.sim.get_trajectory()
        walk_path_2D_dict = self.conver1Dto2D(walk_path_1D)

        # check Max y_length
        y_position_list = []
        for y_list in walk_path_2D_dict.values():
            for y in y_list:
                y_position_list.append(y)
        y_position_list = list(set(y_position_list))
        max_y_length = len(y_position_list) -1

        for x in walk_path_2D_dict:
            diff = max(walk_path_2D_dict[x]) - min(walk_path_2D_dict[x])
            if diff > max_y_length:
                diff = max_y_length
            temp_area = diff -1
            if temp_area > 0:
                area = area + temp_area

        return area


    def get_obs(self):

        ## ray test
        config_map_temp = self.sim.get_state_t_map()
        canvas_map_temp = self.sim.get_canvas_map()
        energy_map_temp = self.sim.get_energy_map()
        defect_map_temp = self.sim.get_defect_map()

        start_point_flag = 0.0
        config_map_temp[self.start_point] = start_point_flag
        canvas_map_temp[self.start_point] = start_point_flag
        energy_map_temp[self.start_point] = start_point_flag
        defect_map_temp[self.start_point] = start_point_flag

        config_map = self._transf2d(config_map_temp)
        canvas_map = self._transf2d(canvas_map_temp)
        energy_map = self._transf2d(energy_map_temp)
        defect_map = self._transf2d(defect_map_temp)

        return np.stack([config_map,
                         canvas_map,
                         energy_map,
                         defect_map
        ], axis=self.stacked_axis)

    @property
    def unwrapped(self):
        """Completely unwrap this env.
            Returns:
                gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def _transf2d(self, s):
        # do we need zero mean here?
        return np.array(s, dtype=np.float32).reshape([self.L, self.L])

    def step_auto(self, action):
        terminate = False
        reward = 0.0
        obs = None
        rets = [0.0, 1.0, 1.0]
        metropolis_executed = False

        '''
            1. detection stage
                if dd & de == 0:
                    flip long loop
                    run metropolis
                if short loop detected:
                    flip short loop
                    run metropolis
                else:
                    walk
            2. reward eval stage
                if metropolis executed:
                    reward with accpetance or not
                if as usual:
                    reward is calculated by dE and dD
        '''
        
        # DETECTION
        if (0<= action < 6):
            # ignore other action idxes
            rets = self.sim.draw(action)
        dE = rets[1]
        dD = rets[2]
            
        if (dE == 0.0 and dD == 0.0):
            self.sim.flip_trajectory()
            rets = self.sim.metropolis()
            metropolis_executed = True
        #TODO: Short loop detection
        
        # EVALUATION

        if (metropolis_executed):
            if rets[0] > 0 and rets[3] > 0:
                print ('ACCEPTS!')
                reward = 1.0
                self.sim.update_config()
                print (self.sim.get_trajectory())
                with open(self.ofilename, 'a') as f:
                    f.write('{}\n'.format(self.sim.get_trajectory()))
                    print ('\tSave loop configuration to file')
                print ('\tTotal accepted number = {}'.format(self.sim.get_updated_counter()))
                print ('\tAccepted loop length = {}'.format(self.sim.get_accepted_length()))
                self.sim.reset()
            else:
                self.sim.reset()
                if (rets[3] > 0):
                    reward = -0.8
                else:
                    reward = -1.0
            # reset or update
        else:
            reward = self._stepwise_weighted_returns(rets)
            # as usual

        # RETURN

        obs = self.get_obs()
        return obs, reward, terminate, rets



