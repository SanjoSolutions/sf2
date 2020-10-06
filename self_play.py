import retro
import os
from baselines.common.retro_wrappers import TimeLimit, wrap_deepmind_retro


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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
        state='ryu_vs_ryu_both_controlled',
        max_episode_steps=99*30,
        scenario=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        obs_type=retro.Observations.IMAGE,  # retro.Observations.RAM,
        players=2,
    )
    env = wrap_deepmind_retro(
        env,
        frame_stack=4
    )
    return env


def main():
    env = make_sf2_env()
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
