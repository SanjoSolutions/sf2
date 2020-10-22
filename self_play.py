import retro
import os
from baselines.common.atari_wrappers import WarpFrame, ScaledFloatFrame
from RyuDiscretizer import RyuDiscretizer
from Runner import Runner
from baselines.common.vec_env.vec_frame_stack import VecFrameStack


FPS = 30

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_sf2_env():
    retro.data.Integrations.add_custom_path(
        os.path.join(SCRIPT_DIR, "custom_integrations")
    )
    env = retro.make(
        game='SuperStreetFighter2-Snes',
        state='ryu_vs_ryu_both_controlled2',
        scenario=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        obs_type=retro.Observations.IMAGE,  # retro.Observations.RAM,
        players=2,
        use_restricted_actions=retro.Actions.FILTERED,
    )
    env = RyuDiscretizer(env)
    # env = WarpFrame(env, grayscale=True)
    # env = ScaledFloatFrame(env)
    return env


def main():
    env = VecFrameStack(make_sf2_env(), 1)
    obs = env.reset()
    n_steps = 128,  # 5 * FPS
    options = {
        'network': 'mlp',  # 'impala_cnn'
        'env': venv,
        'total_timesteps': 40000000,
        'nsteps': n_steps,  # 5 * FPS,  # TODO: Do we still need to pass nsteps here?
        'q_coef': 1.0,
        'ent_coef': 0.001,
        'max_grad_norm': 10,
        'lr': 7e-4,
        'lrschedule': 'linear',
        'rprop_epsilon': 1e-5,
        'rprop_alpha': 0.99,
        'gamma': 0.99,
        'log_interval': 1000,
        'buffer_size': 50000,
        'replay_ratio': 4,
        'replay_start': 10000,
        'c': 10.0,
        'trust_region': True,
        'delta': 1,
        'alpha': 0.99,
        # 'load_path': MODEL_PATH,
        'save_interval': 1000,
        # neuronal network parameters
        'activation': tf.nn.relu,
        'num_layers': 2,  # 4, 2
        'num_hidden': 48,  # 64, 64
        'layer_norm': False,
    }
    models = (
        Acer(**options),
        Acer(**options)
    )
    runner = Runner(env, models, n_steps)
    while True:
        runner.run()
        # obs, rew, done, info = env.step((
        #     env.action_space.sample(),
        #     env.action_space.sample()
        # ))
        # env.render()
        # if done:
        #     obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
