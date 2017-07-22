# -*- coding: utf-8 -*-

LOCAL_T_MAX = 20 # repeat step size
# now, we can not use repeat step?

RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
CHECKPOINT_DIR = 'checkpoints'
LOG_FILE = 'tmp/a3c_log'
INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2   # log_uniform high limit for learning rate

PARALLEL_SIZE = 8 # parallel thread size

ACTION_list = [0, 1, 2, 3, 4, 5, 6] # have 6
# ACTION_list = [0, 1, 2, 3, 4, 5] # auto 6
ACTION_SIZE = len(ACTION_list) # action size

INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.99 # discount factor for rewards
ENTROPY_BETA = 0.01 # entropy regurarlization constant
MAX_TIME_STEP = PARALLEL_SIZE * 10 * 10**7
GRAD_NORM_CLIP = 40.0 # gradient norm clipping
USE_GPU = True # To use GPU, set True
USE_LSTM = True # True for A3C LSTM, False for A3C FF
