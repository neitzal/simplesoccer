import math

import torch

from simplesoccer.mini_env_states import SoccerStates
from simplesoccer.policies.common import EvalPolicy
from simplesoccer.simple_soccer import SimpleSoccer


class HandcraftedPolicy(EvalPolicy):
    def __init__(self, device,
                 n_players,
                 h_corr_dist=3.0,
                 h_corr_dy_shift=1.0,
                 h_corr_dy_lscale=0.1,
                 h_corr_dx_lscale=5.0,
                 v_corr_dist=1.8,
                 v_corr_dx_lscale=1.0,
                 aim_corr_factor=0.2,
                 distance_order_ypos_weight=0.2):
        self.dummy_env = SimpleSoccer(device, opponent_policy=None)

        self.device = device

        self.pi = torch.tensor(math.pi, dtype=torch.float32, device=self.device)
        self.our_goal_pos = torch.tensor([0.0, -self.dummy_env.halffield_height],
                                         dtype=torch.float32, device=self.device)
        self.viz_elems = []


        self.h_corr_dist = h_corr_dist
        self.h_corr_dy_shift = h_corr_dy_shift
        self.h_corr_dy_lscale = h_corr_dy_lscale
        self.h_corr_dx_lscale = h_corr_dx_lscale

        self.v_corr_dist = v_corr_dist
        self.v_corr_dx_lscale = v_corr_dx_lscale

        self.aim_corr_factor = aim_corr_factor

        self.distance_order_ypos_weight = distance_order_ypos_weight
        self.aranges = dict()  # cache

        self.n_players = n_players

    def compute_actions(self, states: SoccerStates):
        """
        :param states: Vectorized states
        :return: Vectorized motion_actions
        """

        assert len(states.batch_shape) == 1
        batch_size = states.batch_shape[0]

        if batch_size in self.aranges:
            arange_batch_size = self.aranges[batch_size]
        else:
            self.aranges[batch_size] = torch.arange(batch_size, device=self.device)
            arange_batch_size = self.aranges[batch_size]

        assert states.objects.shape[1] == self.n_players * 2 + 1

        ball_pos = states.objects[..., 0, :, 0]  # [B, XY]
        ours_pos = states.objects[..., 1:1+self.n_players, :, 0]  # [B, K, XY]

        ball_delta_pos = ball_pos[..., None, :] - ours_pos

        ball_distances = ball_delta_pos.norm(dim=-1)

        # Determine player roles:
        #  ball distance has highest precedence
        #  y-position has second highest precedence
        distance_order = torch.argsort(ball_distances - self.distance_order_ypos_weight * ours_pos[..., 1], dim=-1)

        assert self.n_players >= 3
        attacker_idxs = distance_order[..., 0]
        # defender_idxs = distance_order[..., 1:-1]
        defender_idxs = distance_order[..., 1]
        keeper_idxs = distance_order[..., -1]

        # Horizontal correction:
        #  If player above ball (dy > -h_corr_dy_shift, dx small), correct the target toward the center
        ours_x = ours_pos[..., 0]
        ball_x = ball_pos[..., 0]

        h_corr = (torch.sigmoid((-ball_delta_pos[..., 1, None] + self.h_corr_dy_shift) / self.h_corr_dy_lscale)
                  * torch.exp(-ball_delta_pos[..., 0, None].abs() / self.h_corr_dx_lscale)
                  * ball_x.sign()[..., None, None]
                  * (1 - (torch.sigmoid(ball_delta_pos[..., 1, None] / self.h_corr_dy_lscale)
                          * torch.exp(-ball_delta_pos[..., 0, None].abs() / self.h_corr_dx_lscale)))  # Term that makes sure that if we are directly behind ball, there is no correction
                  * torch.tensor([-self.h_corr_dist, 0.0], device=self.device))

        # Vertical correction:
        #  If abs(dx) large, our target shifts below the ball
        v_corr = ((1 - torch.exp(-ball_delta_pos[..., 0, None].abs() / self.v_corr_dx_lscale))
                  * torch.tensor([0.0, -self.v_corr_dist], device=self.device))

        # Aim correction
        aim_corr = (torch.sigmoid(ball_delta_pos[..., 1, None] / self.h_corr_dy_lscale)
                    * self.aim_corr_factor*ball_x[..., None, None]
                    * torch.tensor([1.0, 0.0], device=self.device))


        target_pos = torch.zeros(states.batch_shape + (self.n_players, 2), device=self.device)

        # Target positions for attackers
        target_pos[arange_batch_size, attacker_idxs, :] = (
                ball_pos
                + h_corr[arange_batch_size, attacker_idxs, :]
                + v_corr[arange_batch_size, attacker_idxs, :]
                + aim_corr[arange_batch_size, attacker_idxs, :]
        )

        # Target positions for defenders: midpoint between goal and ball
        target_pos[arange_batch_size, defender_idxs, :] = (ball_pos + self.our_goal_pos) / 2

        # Target positions for keepers: between goal and ball, but closer to goal
        target_pos[arange_batch_size, keeper_idxs, :] = (ball_pos * 0.2 + self.our_goal_pos * 0.8)

        target_delta_pos = target_pos - ours_pos
        ball_angle = torch.atan2(target_delta_pos[..., 1], target_delta_pos[..., 0])

        discrete_angle = ((((ball_angle + self.pi + 2*self.pi/16) / (2*self.pi/8))) % 8).floor().to(int)

        discrete_angle_to_action = torch.tensor([1, 7, 3, 6, 0, 4, 2, 5, 1], device=self.device)
        motion_actions = discrete_angle_to_action[discrete_angle]

        kick_actions = (ball_delta_pos[..., :, 1] > 0)

        action_template = torch.zeros(states.batch_shape + (self.n_players * 3,),
                                      dtype=torch.int64, device=self.device)


        action_template[..., ::3] = motion_actions
        action_template[arange_batch_size, 3*attacker_idxs + 1] = 1  # attacker dashes
        action_template[..., 2::3] = 2*kick_actions

        return action_template
