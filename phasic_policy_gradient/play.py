import torch
import gym3
from gym3 import types_np
from procgen import ProcgenGym3Env
from . import torch_util as tu
from .tree_util import tree_map

if __name__ == '__main__':
    model = torch.load('/home/breakds/tmp/phasic/ppo_bossfight/model.jd')
    env = ProcgenGym3Env(num=1, env_name='bossfight', render_mode='rgb_array')
    env = gym3.ExtractDictObWrapper(env, "rgb")
    env = gym3.ViewerWrapper(env, info_key="rgb")
    step = 0
    state_in = model.initial_state(env.num)
    print(f'state_in={state_in}')
    while True:
        rew, obs, first = tree_map(tu.np2th, env.observe())        
        action = types_np.sample(env.ac_space, bshape=(env.num,))
        action, state_out, _ = model.act(obs, first, state_in)
        env.act(tree_map(tu.th2np, action))
        state_in = state_out
        print(f"step {step} reward {rew} first {first}")
        if step > 0 and first:
            step = 0
        step += 1
