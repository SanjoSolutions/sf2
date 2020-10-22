"""
Train an agent using Proximal Policy Optimization from OpenAI Baselines
"""

import argparse

import sys
import retro
import os
import numpy as np
import gym
import tensorflow as tf
from baselines.common.vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from ppo2 import ppo2
from baselines.common.atari_wrappers import WarpFrame, ScaledFloatFrame
from acer import acer
from RyuDiscretizer import RyuDiscretizer, RyuDiscretizerDefending


FPS = 30

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SCRIPT_DIR, 'model')
CHECKPOINTS_PATH = os.path.join(LOG_PATH, 'checkpoints')
MODEL_PATH = os.path.join(CHECKPOINTS_PATH, 'latest')


def make_sf2_env():
    retro.data.Integrations.add_custom_path(
        os.path.join(SCRIPT_DIR, "custom_integrations")
    )
    env = retro.make(
        game='SuperStreetFighter2-Snes',
        state=retro.State.DEFAULT,
        scenario=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        obs_type=retro.Observations.RAM,  # retro.Observations.IMAGE
        players=1,  # players=2
        use_restricted_actions=retro.Actions.FILTERED,  # retro.Actions.DISCRETE
    )
    env = RyuDiscretizerDefending(env)
    # env = WarpFrame(env, width=61, height=47, grayscale=True)
    # env = ScaledFloatFrame(env)
    return env


def main():
    os.environ['OPENAI_LOGDIR'] = LOG_PATH

    number_of_environments = 1
    venv = SubprocVecEnv([make_sf2_env] * number_of_environments)
    video_path = './recording'
    video_length = 5 * 60 * FPS
    venv = VecVideoRecorder(venv, video_path, record_video_trigger=lambda step: step %
                            video_length == 0, video_length=video_length)
    # ppo2.learn(
    #     network='mlp',
    #     env=venv,
    #     # eval_env=venv,
    #     total_timesteps=40000000,
    #     nsteps=128,  # 5 * FPS,
    #     nminibatches=number_of_environments,
    #     lam=0.95,
    #     gamma=0.99,
    #     noptepochs=3,
    #     log_interval=1000,
    #     ent_coef=.01,
    #     lr=lambda alpha: 2.5e-4 * alpha,
    #     vf_coef=1.0,
    #     cliprange=lambda alpha: 0.1 * alpha,
    #     save_interval=1000,
    #     # load_path=MODEL_PATH,
    #     # neuronal network parameters
    #     activation=tf.nn.relu,
    #     num_layers=2,  # 4, 2
    #     num_hidden=48,  # 64, 64
    #     layer_norm=False
    # )

    acer.learn(
        network='mlp',  # 'impala_cnn'
        env=venv,
        total_timesteps=40000000,
        nsteps=128,  # 5 * FPS,
        q_coef=1.0,
        ent_coef=0.001,
        max_grad_norm=10,
        lr=7e-4,
        lrschedule='linear',
        rprop_epsilon=1e-5,
        rprop_alpha=0.99,
        gamma=0.99,
        log_interval=1000,
        buffer_size=50000,
        replay_ratio=4,
        replay_start=10000,
        c=10.0,
        trust_region=True,
        delta=1,
        alpha=0.99,
        # load_path=MODEL_PATH,
        save_interval=1000,
        # neuronal network parameters
        activation=tf.nn.relu,
        num_layers=2,  # 4, 2
        num_hidden=48,  # 64, 64
        layer_norm=False
    )


if __name__ == '__main__':
    main()
