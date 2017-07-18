
# Some results

## experiment 7

* train to 617 episodes
* have longer length, larger area loops than past experiments

## experiment 8

* train to 1487 episodes (training faster than exp7)
* use fewer steps to get ACCEPTED loop
* but don't have larger area loop
* setting: `replay_memory_size=500000` , would use 93.1% of 32G memory

## note

* environment setting: [envs](https://github.com/thisray/icegameRL/tree/master/icegame/gym-icegame/gym_icegame/envs)
* `exp 8` add: (1) auto_6_reward (2) more penaltys when take action_6 on unaccepted loop
