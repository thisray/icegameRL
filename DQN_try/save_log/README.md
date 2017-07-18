
# Some results

## experiment 7

* exp7_log_0717_2250.txt
* train to 617 episodes (319,623 steps)
* accepted loops have longer length and larger area than past experiments

## experiment 8

* exp8_log_0718_2320.txt
* train to 1487 episodes (335,396 steps, training faster than exp7)
* use fewer steps to get accepted loop
* but don't have larger area loop
* setting: `replay_memory_size=500000` , would use 95.1% of 32G memory

## note

* environment setting: [envs](https://github.com/thisray/icegameRL/tree/master/icegame/gym-icegame/gym_icegame/envs)
* `exp 8` add: (1) auto_6_reward (2) more penaltys when take action_6 on unaccepted loop
