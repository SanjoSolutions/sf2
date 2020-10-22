import gym
import numpy as np


class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.

    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(
            len(self._decode_discrete_action))

    def action(self, act):
        if isinstance(act, tuple) or isinstance(act, list):
            number_of_actions = int(self.env.action_space.n / self.env.players)
            return np.concatenate(
                tuple(self._decode_discrete_action[act[index]][0:number_of_actions] for index in range(len(act))), axis=None)
        else:
            return self._decode_discrete_action[act].copy()
