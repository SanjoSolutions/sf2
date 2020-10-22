import numpy as np
from baselines.common.runners import AbstractEnvRunner
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from gym import spaces


class Runner(AbstractEnvRunner):

    def __init__(self, env, model, nsteps):
        super().__init__(env=env, model=model, nsteps=nsteps)

        self.nact = env.action_space.n
        nenv = self.nenv
        self.nbatch = nenv * nsteps
        self.batch_ob_shape = (nenv*(nsteps+1),) + env.observation_space.shape

        self.obs = env.reset()
        self.obs_dtype = env.observation_space.dtype
        self.ac_dtype = env.action_space.dtype
        self.nstack = self.env.nstack
        self.nc = self.batch_ob_shape[-1] // self.nstack

    def run(self):
        runs = tuple(create_run() for index in range(len(self.model))
        enc_obs=np.split(self.env.stackedobs, self.env.nstack, axis=-1)
        for _ in range(self.nsteps):
            actions=[None * len(self.model)]
            for index in range(len(self.model)):
              run=runs[index]
              model=self.model[index]
              action, mus, states=model._step(
                  self.obs, S=self.states, M=self.dones)
              actions[index]=action
              run['obs'].append(np.copy(self.obs))
              run['actions'].append(actions)
              run['mus'].append(mus)
              run['dones'].append(self.dones)

            obs, rewards, dones, _=self.env.step(actions)

            # states information for statefull models like LSTM
            self.states=states
            self.dones=dones
            self.obs=obs

            mb_rewards.append(rewards)
            enc_obs.append(obs[..., -self.nc:])
        mb_obs.append(np.copy(self.obs))
        mb_dones.append(self.dones)

        enc_obs=np.asarray(enc_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_obs=np.asarray(mb_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_actions=np.asarray(mb_actions, dtype=self.ac_dtype).swapaxes(1, 0)
        mb_rewards=np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_mus=np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)

        mb_dones=np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        mb_masks=mb_dones  # Used for statefull models like LSTM's to mask state when done
        # Used for calculating returns. The dones array is now aligned with rewards
        mb_dones=mb_dones[:, 1:]

        # shapes are now [nenv, nsteps, []]
        # When pulling from buffer, arrays will now be reshaped in place, preventing a deep copy.

        # enc_obs, obs, actions, rewards, mus, dones, masks
        # return enc_obs, mb_obs, mb_actions, mb_rewards, mb_mus, mb_dones, mb_masks

def create_run():
  run={}
  run['obs']=[]
  run['actions']=[]
  run['mus']=[]
  run['dones']=[]
  run['rewards']=[]
  return run
