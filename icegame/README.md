# icegame (from [kvzhao](https://github.com/kvzhao/icegame))
Designed as an environments interacting with spin ice system.
Original enviroment and Cpp code from: https://github.com/kvzhao/icegame


## Intro
This dir only save gym-icegame enviroment code (`icegame_env.py`).

## about Version
I use the original version of `libicegame.so`, which don't have `sim.clear_buffer()` and I still use `sim.reset()` now.

## Install gym-icegame

after compile the C++ code

```
python setup.py install
```

which depends on openai gym.
