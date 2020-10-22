from Discretizer import Discretizer


class RyuDiscretizer(Discretizer):
    def __init__(self, env):
        LP = 'Y'
        MP = 'X'
        HP = 'L'
        LK = 'B'
        MK = 'A'
        HK = 'R'
        CROUCH = 'DOWN'
        JUMP = 'UP'
        super().__init__(env=env, combos=[
            # Moves shared by all characters
            [],
            ['LEFT'],
            ['RIGHT'],
            [CROUCH],
            [JUMP],
            [CROUCH, 'LEFT'],
            [CROUCH, 'RIGHT'],
            [JUMP, 'LEFT'],
            [JUMP, 'RIGHT'],
            [LP],
            [MP],
            [HP],
            [LK],
            [MK],
            [HK],
            [CROUCH, LP],
            [CROUCH, MP],
            [CROUCH, HP],
            [CROUCH, LK],
            [CROUCH, MK],
            [CROUCH, HK],
            # Ryu moves
            ['LEFT', MP],
            ['RIGHT', MP],
            ['LEFT', HP],
            ['RIGHT', HP],
            ['LEFT', MK],
            ['RIGHT', MK],
            ['LEFT', HK],
            ['RIGHT', HK],
            ['LEFT', LP],
            ['RIGHT', LP],
            ['LEFT', LK],
            ['RIGHT', LK],
        ])


class RyuDiscretizerDefending(Discretizer):
    def __init__(self, env):
        LP = 'Y'
        MP = 'X'
        HP = 'L'
        LK = 'B'
        MK = 'A'
        HK = 'R'
        CROUCH = 'DOWN'
        JUMP = 'UP'
        super().__init__(env=env, combos=[
            [CROUCH, 'LEFT'],
            [CROUCH, 'RIGHT'],
            ['LEFT'],
            ['RIGHT'],
            ['LEFT', MP],
            ['RIGHT', MP],
        ])
