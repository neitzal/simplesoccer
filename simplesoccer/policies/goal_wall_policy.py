import math

import torch

from simplesoccer.mini_env_states import SoccerStates
from simplesoccer.policies.common import EvalPolicy
from simplesoccer.simple_soccer import SimpleSoccer


class GoalWallPolicy(EvalPolicy):

    def __init__(self, device, n_players, visualize=False):
        dummy_env = SimpleSoccer(device, opponent_policy=None)

        self.device = device

        self.pi = torch.tensor(math.pi, dtype=torch.float32, device=self.device)
        self.our_goal_pos = torch.tensor([0.0, -dummy_env.halffield_height],
                                         dtype=torch.float32, device=self.device)
        self.goal_halfwidth = dummy_env.goal_width / 2
        left = -self.goal_halfwidth + dummy_env.player_radius
        target_x = torch.linspace(left, -left, steps=n_players, device=device)
        target_y = -dummy_env.halffield_height + 1.5
        target_y = torch.full(size=(n_players,), fill_value=target_y, device=device)
        self.target_pos = torch.column_stack((target_x, target_y))

        self.visualize = visualize
        self.viz_elems = []

        self.n_players = n_players

    def compute_actions(self, states: SoccerStates):
        """
        :param states: Vectorized states
        :return: Vectorized motion_actions
        """
        assert len(states.batch_shape) == 1
        n_players = self.n_players
        assert states.objects.shape[1] == n_players * 2 + 1

        ball_pos = states.objects[..., 0, :, 0]  # [B, XY]
        ours_pos = states.objects[..., 1:1 + n_players, :, 0]  # [B, K, XY]

        ball_delta_pos = ball_pos[..., None, :] - ours_pos

        # Get ranks of players according to their x position, such that
        # they don't have to cross when moving toward their target positions
        ranks = torch.argsort(torch.argsort(ours_pos[..., 0], dim=-1), dim=-1)
        target_pos = self.target_pos[ranks]

        target_delta_pos = target_pos - ours_pos
        ball_angle = torch.atan2(target_delta_pos[..., 1], target_delta_pos[..., 0])

        discrete_angle = ((ball_angle + self.pi + 2 * self.pi / 16 / 2 * self.pi / 8) % 8).floor().to(int)

        discrete_angle_to_action = torch.tensor([1, 7, 3, 6, 0, 4, 2, 5, 1], device=self.device)
        motion_actions = discrete_angle_to_action[discrete_angle]

        kick_actions = (ball_delta_pos[..., :, 1] > 0)
        action_template = torch.zeros(states.batch_shape + (n_players * 3,), dtype=torch.int64, device=self.device)

        action_template[..., ::3] = motion_actions
        action_template[..., 1::3] = 1  # everyone dashes
        action_template[..., 2::3] = 2 * kick_actions

        return action_template
