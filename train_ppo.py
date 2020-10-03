"""
Train an agent using Proximal Policy Optimization from OpenAI Baselines
"""

import argparse

import sys
import retro
import os
import numpy as np
import gym
from baselines.common.vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from ppo2 import ppo2
from baselines.common.retro_wrappers import TimeLimit, wrap_deepmind_retro


FPS = 30

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SCRIPT_DIR, 'model')
CHECKPOINTS_PATH = os.path.join(LOG_PATH, 'checkpoints')
MODEL_PATH = os.path.join(CHECKPOINTS_PATH, 'latest')


class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        done = False
        totrew = 0 if self.env.players <= 1 else [0] * self.env.players
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n-1:
                ob, rew, done, info = self.env.step(
                    self.curac, want_render=False)
            else:
                ob, rew, done, info = self.env.step(self.curac)
            if isinstance(totrew, list):
                for index in range(len(min(totrew, rew))):
                    totrew[index] += rew[index]
            else:
                totrew += rew
            if done:
                break
        return ob, totrew, done, info

    def seed(self, s):
        self.rng.seed(s)


def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    import retro
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs)
    # env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def make_sf2_env():
    retro.data.Integrations.add_custom_path(
        os.path.join(SCRIPT_DIR, "custom_integrations")
    )
    env = make_retro(
        game='SuperStreetFighter2-Snes',
        state=retro.State.DEFAULT,
        scenario=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        obs_type=retro.Observations.IMAGE,  # retro.Observations.RAM,
        players=1,  # players=2
    )
    env = wrap_deepmind_retro(
        env,
        frame_stack=4
    )
    return env


def main():
    os.environ['OPENAI_LOGDIR'] = LOG_PATH

    number_of_environments = 1
    venv = SubprocVecEnv([make_sf2_env] * number_of_environments)
    video_path = './recording'
    video_length = 5 * 60 * FPS
    venv = VecVideoRecorder(venv, video_path, record_video_trigger=lambda step: step %
                            video_length == 0, video_length=video_length)
    ppo2.learn(
        network='cnn_lstm',
        env=venv,
        eval_env=venv,
        total_timesteps=int(sys.maxsize),
        nsteps=10 * FPS,
        nminibatches=number_of_environments,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        ent_coef=.01,
        lr=lambda f: f * 2.5e-4,
        cliprange=0.1,
        save_interval=10,
        # load_path=MODEL_PATH,
    )


if __name__ == '__main__':
    main()
