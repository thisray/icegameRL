## A3C test - 2
I think this is better than `A3C_try_1`.

## Link
* kvzhao/rlloop: https://github.com/kvzhao/rlloop
* dennybritz/reinforcement-learning: https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient/a3c

## Components
* `train.py`: contains the main method to start training.
* `estimators.py`: contains the Tensorflow graph definitions for the Policy and Value networks.
* `worker.py`: contains code that runs in each worker threads.
* `policy_monitor.py`: contains code that evaluates the policy network by running an episode and saving rewards to Tensorboard.

