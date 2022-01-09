"""
Collection of initial states that are meant to check the settings
of physical parameters of the environment.
"""

import math
import subprocess
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import moviepy.editor as mpy
import numpy as np
import torch

from simplesoccer.simple_soccer import SimpleSoccer
from simplesoccer.torch_vec_env import TorchVecEnv

ObjectState = namedtuple('ObjectState', ['x', 'y', 'vx', 'vy'])

STRESS_TESTS = [
    {
        'objects': {
            'ball': ObjectState(x=0, y=0, vx=0, vy=0),
            'p1_1': ObjectState(x=-2, y=-2, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=-2, vx=0, vy=0),
            'p1_3': ObjectState(x=2, y=-2, vx=0, vy=0),
            'p2_1': ObjectState(x=2, y=2, vx=0, vy=0),
            'p2_2': ObjectState(x=0, y=2, vx=0, vy=0),
            'p2_3': ObjectState(x=-2, y=2, vx=0, vy=0),
        },
        'our_actions': [1, 0, 0, 1, 0, 0, 1, 0, 0],
        'opp_actions': [1, 0, 0, 1, 0, 0, 1, 0, 0],
    },
    {
        'objects': {
            'ball': ObjectState(x=0, y=0, vx=0, vy=0),
            'p1_1': ObjectState(x=-1.5, y=-2, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=-2.75, vx=0, vy=0),
            'p1_3': ObjectState(x=1.5, y=-2, vx=0, vy=0),
            'p2_1': ObjectState(x=2, y=2, vx=0, vy=0),
            'p2_2': ObjectState(x=0, y=2, vx=0, vy=0),
            'p2_3': ObjectState(x=-2, y=2, vx=0, vy=0),
        },
        'our_actions': [2, 0, 0, 2, 0, 0, 2, 0, 0],
        'opp_actions': [2, 0, 0, 2, 0, 0, 2, 0, 0],
    },
    {
        'objects': {
            'ball': ObjectState(x=0, y=0, vx=0, vy=0),
            'p1_1': ObjectState(x=-1.5, y=-2, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=-2.75, vx=0, vy=0),
            'p1_3': ObjectState(x=1.5, y=-2, vx=0, vy=0),
            'p2_1': ObjectState(x=1.5, y=2, vx=0, vy=0),
            'p2_2': ObjectState(x=0, y=2.75, vx=0, vy=0),
            'p2_3': ObjectState(x=-1.5, y=2, vx=0, vy=0),
        },
        'our_actions': [2, 0, 0, 2, 0, 0, 2, 0, 0],
        'opp_actions': [2, 0, 0, 2, 0, 0, 2, 0, 0],
    },
    {  # Ball tries to tunnel straight through player
        'objects': {
            'ball': ObjectState(x=0, y=0, vx=0, vy=-50),
            'p1_1': ObjectState(x=-3.5, y=-3, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=-3, vx=0, vy=0),
            'p1_3': ObjectState(x=3.5, y=-3, vx=0, vy=0),
            'p2_1': ObjectState(x=3.5, y=3, vx=0, vy=0),
            'p2_2': ObjectState(x=0, y=3, vx=0, vy=0),
            'p2_3': ObjectState(x=-3.5, y=3, vx=0, vy=0),
        },
        'our_actions': [1, 0, 0, 2, 0, 0, 0, 0, 0],
        'opp_actions': [1, 0, 0, 2, 0, 0, 0, 0, 0],
    },
    {  # Ball tries to tunnel straight through player
        'objects': {
            'ball': ObjectState(x=0, y=0, vx=0, vy=-50),
            'p1_1': ObjectState(x=-3.5, y=-3, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=-2, vx=0, vy=10),
            'p1_3': ObjectState(x=3.5, y=-3, vx=0, vy=0),
            'p2_1': ObjectState(x=3.5, y=3, vx=0, vy=0),
            'p2_2': ObjectState(x=0, y=3, vx=0, vy=0),
            'p2_3': ObjectState(x=-3.5, y=3, vx=0, vy=0),
        },
        'our_actions': [2, 1, 0, 2, 1, 0, 2, 1, 0],
        'opp_actions': [1, 0, 0, 2, 1, 0, 0, 0, 0],
    },
    {  # Ball tries to tunnel straight through player (slightly displaced)
        'objects': {
            'ball': ObjectState(x=0.01, y=0, vx=1, vy=-50),
            'p1_1': ObjectState(x=-3.5, y=-3, vx=0, vy=0),
            'p1_2': ObjectState(x=0.05, y=-2, vx=0, vy=10),
            'p1_3': ObjectState(x=3.5, y=-3, vx=0, vy=0),
            'p2_1': ObjectState(x=3.5, y=3, vx=0, vy=0),
            'p2_2': ObjectState(x=0, y=3, vx=0, vy=0),
            'p2_3': ObjectState(x=-3.5, y=3, vx=0, vy=0),
        },
        'our_actions': [2, 1, 0, 2, 1, 0, 2, 1, 0],
        'opp_actions': [1, 0, 0, 2, 1, 0, 0, 0, 0],
    },
    {  # Ball tries to tunnel straight through player
        'objects': {
            'ball': ObjectState(x=0, y=0, vx=0, vy=-50),
            'p1_1': ObjectState(x=-3.5, y=-3, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=-2.1, vx=0, vy=10),
            'p1_3': ObjectState(x=3.5, y=-3, vx=0, vy=0),
            'p2_1': ObjectState(x=3.5, y=3, vx=0, vy=0),
            'p2_2': ObjectState(x=0, y=3, vx=0, vy=0),
            'p2_3': ObjectState(x=-3.5, y=3, vx=0, vy=0),
        },
        'our_actions': [2, 1, 0, 2, 1, 0, 2, 1, 0],
        'opp_actions': [1, 0, 0, 2, 0, 0, 0, 0, 0],
    },
    {  # Player kicks ball on opponent
        'objects': {
            'ball': ObjectState(x=0, y=0, vx=0, vy=0),
            'p1_1': ObjectState(x=-3.5, y=-3, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=-1.0, vx=0, vy=0),
            'p1_3': ObjectState(x=3.5, y=-3, vx=0, vy=0),
            'p2_1': ObjectState(x=3.5, y=3, vx=0, vy=0),
            'p2_2': ObjectState(x=0, y=3, vx=0, vy=0),
            'p2_3': ObjectState(x=-3.5, y=3, vx=0, vy=0),
        },
        'our_actions': [8, 0, 0, 8, 0, 2, 8, 1, 0],
        'opp_actions': [8, 0, 0, 8, 0, 0, 8, 0, 0],
    },
    {
        'objects': {
            'ball': ObjectState(x=0.75, y=-1, vx=0, vy=-20),
            'p1_1': ObjectState(x=-3.5, y=-3, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=-3, vx=0, vy=0),
            'p1_3': ObjectState(x=3.5, y=-3, vx=0, vy=0),
            'p2_1': ObjectState(x=3.5, y=3, vx=0, vy=0),
            'p2_2': ObjectState(x=0, y=3, vx=0, vy=0),
            'p2_3': ObjectState(x=-3.5, y=3, vx=0, vy=0),
        },
        'our_actions': [1, 0, 0, 2, 0, 0, 0, 0, 0],
        'opp_actions': [1, 0, 0, 2, 0, 0, 0, 0, 0],
    },
    {
        'objects': {
            'ball': ObjectState(x=0.65, y=-1, vx=0, vy=-20),
            'p1_1': ObjectState(x=-3.5, y=-3, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=-3, vx=0, vy=0),
            'p1_3': ObjectState(x=3.5, y=-3, vx=0, vy=0),
            'p2_1': ObjectState(x=3.5, y=3, vx=0, vy=0),
            'p2_2': ObjectState(x=0, y=3, vx=0, vy=0),
            'p2_3': ObjectState(x=-3.5, y=3, vx=0, vy=0),
        },
        'our_actions': [1, 0, 0, 2, 0, 0, 0, 0, 0],
        'opp_actions': [1, 0, 0, 2, 0, 0, 0, 0, 0],
    },
    {
        'objects': {
            'ball': ObjectState(x=0.6, y=-1, vx=0, vy=-20),
            'p1_1': ObjectState(x=-3.5, y=-3, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=-3, vx=0, vy=0),
            'p1_3': ObjectState(x=3.5, y=-3, vx=0, vy=0),
            'p2_1': ObjectState(x=3.5, y=3, vx=0, vy=0),
            'p2_2': ObjectState(x=0, y=3, vx=0, vy=0),
            'p2_3': ObjectState(x=-3.5, y=3, vx=0, vy=0),
        },
        'our_actions': [1, 0, 0, 2, 0, 0, 0, 0, 0],
        'opp_actions': [1, 0, 0, 2, 0, 0, 0, 0, 0],
    },
    {
        'objects': {
            'ball': ObjectState(x=0.6, y=-1, vx=0, vy=0),
            'p1_1': ObjectState(x=-3.5, y=-3, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=-3, vx=0, vy=0),
            'p1_3': ObjectState(x=4.5, y=3, vx=0, vy=0),
            'p2_1': ObjectState(x=-3.5, y=-3 + 1.75, vx=0, vy=0),
            'p2_2': ObjectState(x=0 + 1.75, y=-3, vx=0, vy=0),
            'p2_3': ObjectState(x=4.5 + 1.2, y=3 + 1.2, vx=0, vy=0),
        },
        'our_actions': [2, 0, 0, 0, 0, 0, 4, 0, 0],
        'opp_actions': [2, 0, 0, 0, 0, 0, 4, 0, 0],
    },
    {
        'objects': {
            'ball': ObjectState(x=0.2, y=-1, vx=0, vy=-10),
            'p1_1': ObjectState(x=-3.5, y=-3, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=-3, vx=0, vy=0),
            'p1_3': ObjectState(x=3.5, y=-3, vx=0, vy=0),
            'p2_1': ObjectState(x=3.5, y=3, vx=0, vy=0),
            'p2_2': ObjectState(x=0, y=3, vx=0, vy=0),
            'p2_3': ObjectState(x=-3.5, y=3, vx=0, vy=0),
        },
        'our_actions': [1, 0, 0, 2, 0, 0, 0, 0, 0],
        'opp_actions': [1, 0, 0, 2, 0, 0, 0, 0, 0],
    },

    # Trying to push the opponent
    {
        'objects': {
            'ball': ObjectState(x=4, y=4, vx=0, vy=0),
            'p1_1': ObjectState(x=-0.8, y=0, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=-1.75, vx=0, vy=0),
            'p1_3': ObjectState(x=0.8, y=0, vx=0, vy=0),
            'p2_1': ObjectState(x=7.5, y=2, vx=0, vy=0),
            'p2_2': ObjectState(x=0, y=1.75, vx=0, vy=0),
            'p2_3': ObjectState(x=-7.5, y=2, vx=0, vy=0),
        },
        'our_actions': [4, 0, 0, 2, 0, 0, 5, 0, 0],
        'opp_actions': [2, 0, 0, 2, 0, 0, 2, 0, 0],
    },

    # Ball moves towards goal, but very fast (might be limited by maximum object vel)
    {
        'objects': {
            'ball': ObjectState(x=0, y=-9.45, vx=0, vy=-100.0),
            'p1_1': ObjectState(x=-0.8, y=0, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=-1.75, vx=0, vy=0),
            'p1_3': ObjectState(x=0.8, y=0, vx=0, vy=0),
            'p2_1': ObjectState(x=7.5, y=2, vx=0, vy=0),
            'p2_2': ObjectState(x=0, y=1.75, vx=0, vy=0),
            'p2_3': ObjectState(x=-7.5, y=2, vx=0, vy=0),
        },
        'our_actions': [4, 0, 0, 2, 0, 0, 5, 0, 0],
        'opp_actions': [2, 0, 0, 2, 0, 0, 2, 0, 0],
    },

    # Player tries to shoot goal ("penalty kick") but without kicking
    {
        'objects': {
            'ball': ObjectState(x=0, y=5., vx=0, vy=0.0),
            'p1_1': ObjectState(x=-6, y=0, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=2.0, vx=0, vy=3.),
            'p1_3': ObjectState(x=6, y=0, vx=0, vy=0),
            'p2_1': ObjectState(x=7.5, y=2, vx=0, vy=0),
            'p2_2': ObjectState(x=4., y=1.75, vx=0, vy=0),
            'p2_3': ObjectState(x=-7.5, y=2, vx=0, vy=0),
        },
        'our_actions': [2, 0, 0, 2, 0, 0, 2, 0, 0],
        'opp_actions': [2, 0, 0, 2, 0, 0, 2, 0, 0],
    },

    # Player tries to shoot goal ("penalty kick")  but without kicking
    {
        'objects': {
            'ball': ObjectState(x=0, y=5.0, vx=0, vy=0.0),
            'p1_1': ObjectState(x=-6, y=0, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=1.9, vx=0, vy=3.),
            'p1_3': ObjectState(x=6, y=0, vx=0, vy=0),
            'p2_1': ObjectState(x=7.5, y=2, vx=0, vy=0),
            'p2_2': ObjectState(x=4., y=1.75, vx=0, vy=0),
            'p2_3': ObjectState(x=-7.5, y=2, vx=0, vy=0),
        },
        'our_actions': [2, 0, 0, 2, 0, 0, 2, 0, 0],
        'opp_actions': [2, 0, 0, 2, 0, 0, 2, 0, 0],
    },
    # Player tries to shoot goal ("penalty kick")  but without kicking
    {
        'objects': {
            'ball': ObjectState(x=0, y=5.0, vx=0, vy=0.0),
            'p1_1': ObjectState(x=-6, y=0, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=1.8, vx=0, vy=3.),
            'p1_3': ObjectState(x=6, y=0, vx=0, vy=0),
            'p2_1': ObjectState(x=7.5, y=2, vx=0, vy=0),
            'p2_2': ObjectState(x=4., y=1.75, vx=0, vy=0),
            'p2_3': ObjectState(x=-7.5, y=2, vx=0, vy=0),
        },
        'our_actions': [2, 0, 0, 2, 0, 0, 2, 0, 0],
        'opp_actions': [2, 0, 0, 2, 0, 0, 2, 0, 0],
    },
    # Player tries to shoot goal ("penalty kick")  but without kicking
    {
        'objects': {
            'ball': ObjectState(x=0, y=5.0, vx=0, vy=0.0),
            'p1_1': ObjectState(x=-6, y=0, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=1.7, vx=0, vy=3.),
            'p1_3': ObjectState(x=6, y=0, vx=0, vy=0),
            'p2_1': ObjectState(x=7.5, y=2, vx=0, vy=0),
            'p2_2': ObjectState(x=4., y=1.75, vx=0, vy=0),
            'p2_3': ObjectState(x=-7.5, y=2, vx=0, vy=0),
        },
        'our_actions': [2, 0, 0, 2, 0, 0, 2, 0, 0],
        'opp_actions': [2, 0, 0, 2, 0, 0, 2, 0, 0],
    },
    # Player tries to shoot goal while dashing ("penalty kick")  but without kicking
    {
        'objects': {
            'ball': ObjectState(x=0, y=5.0, vx=0, vy=0.0),
            'p1_1': ObjectState(x=-6, y=0, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=0., vx=0, vy=3.),
            'p1_3': ObjectState(x=6, y=0, vx=0, vy=0),
            'p2_1': ObjectState(x=7.5, y=2, vx=0, vy=0),
            'p2_2': ObjectState(x=4., y=1.75, vx=0, vy=0),
            'p2_3': ObjectState(x=-7.5, y=2, vx=0, vy=0),
        },
        'our_actions': [2, 0, 0, 2, 1, 0, 2, 0, 0],
        'opp_actions': [2, 0, 0, 2, 0, 0, 2, 0, 0],
    },
    # Player tries to shoot goal ("penalty kick") but without kicking, with slight displacement
    {
        'objects': {
            'ball': ObjectState(x=0, y=5.0, vx=0, vy=0.0),
            'p1_1': ObjectState(x=-6, y=0, vx=0, vy=0),
            'p1_2': ObjectState(x=0.1, y=1.7, vx=0, vy=3.),
            'p1_3': ObjectState(x=6, y=0, vx=0, vy=0),
            'p2_1': ObjectState(x=7.5, y=2, vx=0, vy=0),
            'p2_2': ObjectState(x=4., y=1.75, vx=0, vy=0),
            'p2_3': ObjectState(x=-7.5, y=2, vx=0, vy=0),
        },
        'our_actions': [2, 0, 0, 2, 0, 0, 2, 0, 0],
        'opp_actions': [2, 0, 0, 2, 0, 0, 2, 0, 0],
    },
    # Player tries to shoot goal ("penalty kick") with slight displacement, including kick
    {
        'objects': {
            'ball': ObjectState(x=0, y=5.0, vx=0, vy=0.0),
            'p1_1': ObjectState(x=-6, y=0, vx=0, vy=0),
            'p1_2': ObjectState(x=0.1, y=1.7, vx=0, vy=3.),
            'p1_3': ObjectState(x=6, y=0, vx=0, vy=0),
            'p2_1': ObjectState(x=7.5, y=2, vx=0, vy=0),
            'p2_2': ObjectState(x=4., y=1.75, vx=0, vy=0),
            'p2_3': ObjectState(x=-7.5, y=2, vx=0, vy=0),
        },
        'our_actions': [2, 0, 0, 2, 0, 2, 2, 0, 0],
        'opp_actions': [2, 0, 0, 2, 0, 0, 2, 0, 0],
    },

    # Players in circle, move inwards
    {
        'objects': {
            'ball': ObjectState(x=0, y=0.0, vx=2., vy=20.0),
            'p1_1': ObjectState(x=-3. / math.sqrt(2), y=-3. / math.sqrt(2), vx=0, vy=0),
            'p1_2': ObjectState(x=0., y=-3., vx=0, vy=0),
            'p1_3': ObjectState(x=+3. / math.sqrt(2), y=-3. / math.sqrt(2), vx=0, vy=0),
            'p2_1': ObjectState(x=-3. / math.sqrt(2), y=+3. / math.sqrt(2), vx=0, vy=0),
            'p2_2': ObjectState(x=0., y=+3., vx=0, vy=0),
            'p2_3': ObjectState(x=+3. / math.sqrt(2), y=+3. / math.sqrt(2), vx=0, vy=0),
        },
        'our_actions': [4, 0, 0, 2, 0, 0, 5, 0, 0],
        'opp_actions': [5, 0, 0, 2, 0, 0, 4, 0, 0],
    },

    # Side-ways shot
    {
        'objects': {
            'ball': ObjectState(x=0, y=0.0, vx=0, vy=0.0),
            'p1_1': ObjectState(x=-6, y=0, vx=0, vy=0),
            'p1_2': ObjectState(x=-2, y=-0.72, vx=0, vy=0.),
            'p1_3': ObjectState(x=6, y=0, vx=0, vy=0),
            'p2_1': ObjectState(x=7.5, y=2, vx=0, vy=0),
            'p2_2': ObjectState(x=4., y=1.75, vx=0, vy=0),
            'p2_3': ObjectState(x=-7.5, y=2, vx=0, vy=0),
        },
        'our_actions': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'opp_actions': [5, 0, 0, 2, 0, 0, 4, 0, 0],
    },
    # How far does ball roll at most if uninterrupted
    {
        'objects': {
            'ball': ObjectState(x=-9.9, y=-9.9, vx=1e4, vy=1e4),
            'p1_1': ObjectState(x=-6, y=-8, vx=0, vy=0),
            'p1_2': ObjectState(x=-4, y=-8, vx=0, vy=0.),
            'p1_3': ObjectState(x=-2, y=-8, vx=0, vy=0),
            'p2_1': ObjectState(x=0, y=-8, vx=0, vy=0),
            'p2_2': ObjectState(x=2., y=-8, vx=0, vy=0),
            'p2_3': ObjectState(x=4., y=-8, vx=0, vy=0),
        },
        'our_actions': [3, 0, 0, 3, 0, 0, 3, 0, 0],
        'opp_actions': [2, 0, 0, 2, 0, 0, 2, 0, 0],
    },

    # Ball rolls towards player (centered)
    {
        'objects': {
            'ball': ObjectState(x=0, y=3.0, vx=0, vy=-15.0),
            'p1_1': ObjectState(x=-6, y=0, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=0.0, vx=0, vy=+1.),
            'p1_3': ObjectState(x=6, y=0, vx=0, vy=0),
            'p2_1': ObjectState(x=7.5, y=2, vx=0, vy=0),
            'p2_2': ObjectState(x=4., y=1.75, vx=0, vy=0),
            'p2_3': ObjectState(x=-7.5, y=2, vx=0, vy=0),
        },
        'our_actions': [8, 0, 0, 8, 0, 0, 8, 0, 0],
        'opp_actions': [8, 0, 0, 8, 0, 0, 8, 0, 0],
    },
    # Ball rolls towards player (centered)
    {
        'objects': {
            'ball': ObjectState(x=0, y=2.9, vx=0, vy=-15.0),
            'p1_1': ObjectState(x=-6, y=0, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=0.0, vx=0, vy=+1.),
            'p1_3': ObjectState(x=6, y=0, vx=0, vy=0),
            'p2_1': ObjectState(x=7.5, y=2, vx=0, vy=0),
            'p2_2': ObjectState(x=4., y=1.75, vx=0, vy=0),
            'p2_3': ObjectState(x=-7.5, y=2, vx=0, vy=0),
        },
        'our_actions': [8, 0, 0, 8, 0, 0, 8, 0, 0],
        'opp_actions': [8, 0, 0, 8, 0, 0, 8, 0, 0],
    },
    # Ball rolls towards player (centered)
    {
        'objects': {
            'ball': ObjectState(x=0.1, y=2.8, vx=0, vy=-15.0),
            'p1_1': ObjectState(x=-6, y=0, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=0.0, vx=0, vy=+1.),
            'p1_3': ObjectState(x=6, y=0, vx=0, vy=0),
            'p2_1': ObjectState(x=7.5, y=2, vx=0, vy=0),
            'p2_2': ObjectState(x=4., y=1.75, vx=0, vy=0),
            'p2_3': ObjectState(x=-7.5, y=2, vx=0, vy=0),
        },
        'our_actions': [8, 0, 0, 8, 0, 0, 8, 0, 0],
        'opp_actions': [8, 0, 0, 8, 0, 0, 8, 0, 0],
    },

    # Ball starts colliding, moves away
    {
        'objects': {
            'ball': ObjectState(x=0, y=0.8, vx=0, vy=+0.1),
            'p1_1': ObjectState(x=-6, y=0, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=0.0, vx=0, vy=-0.42),
            'p1_3': ObjectState(x=6, y=0, vx=0, vy=0),
            'p2_1': ObjectState(x=7.5, y=2, vx=0, vy=0),
            'p2_2': ObjectState(x=4., y=1.75, vx=0, vy=0),
            'p2_3': ObjectState(x=-7.5, y=2, vx=0, vy=0),
        },
        'our_actions': [8, 0, 0, 8, 0, 0, 8, 0, 0],
        'opp_actions': [8, 0, 0, 8, 0, 0, 8, 0, 0],
    },
    # Ball lies close to player, player kicks
    {
        'objects': {
            'ball': ObjectState(x=0.74, y=0.74, vx=0, vy=0),
            'p1_1': ObjectState(x=-6, y=0, vx=0, vy=0),
            'p1_2': ObjectState(x=0, y=0.0, vx=0, vy=0.0),
            'p1_3': ObjectState(x=6, y=0, vx=0, vy=0),
            'p2_1': ObjectState(x=7.5, y=2, vx=0, vy=0),
            'p2_2': ObjectState(x=4., y=1.75, vx=0, vy=0),
            'p2_3': ObjectState(x=-7.5, y=2, vx=0, vy=0),
        },
        'our_actions': [8, 0, 0, 8, 0, 2, 8, 0, 0],
        'opp_actions': [8, 0, 0, 8, 0, 0, 8, 0, 0],
    },

]


def objects_to_state(objects):
    #  [envs, objects, xy, pos/vel]
    object_ids = ['ball']
    for team in [1, 2]:
        for player in [1, 2, 3]:
            object_ids.append(f'p{team}_{player}')
    state = []
    for object_id in object_ids:
        state.append([[objects[object_id].x, objects[object_id].vx],
                      [objects[object_id].y, objects[object_id].vy]])
    return torch.tensor(state, dtype=torch.float32)


class ConstantPolicy:
    def __init__(self, actions):
        self.actions = torch.tensor(actions)

    def forward(self, *args, **kwargs):
        return self.actions, None, None, None


def stress_test():
    torch.manual_seed(123)
    np.random.seed(123)

    device = 'cpu'
    model = SimpleSoccer(device, opponent_policy=None)

    num_envs = len(STRESS_TESTS)

    neural_venv = TorchVecEnv(model, num_envs=num_envs, device=device,
                              render_n_envs=len(STRESS_TESTS))
    neural_venv.reset()

    actions = []
    opponent_actions = []
    for i, stresstest in enumerate(STRESS_TESTS):
        neural_venv.current_states.objects[i] = objects_to_state(stresstest['objects'])
        actions.append(stresstest['our_actions'])
        opponent_actions.append(stresstest['opp_actions'])

    actions = np.array(actions)
    opponent_actions = np.array(opponent_actions)

    model.opponent_policy = ConstantPolicy(opponent_actions)

    imgs = [neural_venv.render('rgb_array')]
    all_obses = []
    for i_step in range(50):
        obses, rewards, dones, info = neural_venv.step(actions)
        all_obses.extend(obses)
        print(f'rewards: {rewards}')
        img = neural_venv.render('rgb_array')
        imgs.append(img)

    clip = mpy.ImageSequenceClip(imgs, fps=int(round(1/model.dt)))

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'mini_env_stress_test_{timestamp}.mp4'
    parent_path = Path(__file__).parent.parent / 'videos' / 'stress_test'
    parent_path.mkdir(exist_ok=True)
    filepath = parent_path / filename
    clip.write_videofile(str(filepath))
    subprocess.run(['mpv', '--loop', '--scale=nearest', '--geometry=1600x1600',
                    str(filepath)])


if __name__ == '__main__':
    stress_test()
