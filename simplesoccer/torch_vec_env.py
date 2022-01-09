from typing import Optional, List, Union, Sequence, Tuple

import numpy as np
import torch

from simplesoccer.utils.base_vec_env import tile_images

TorchVecEnvStepReturn = Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                              Tuple[torch.Tensor, torch.Tensor]]


class TorchVecEnv:
    def __init__(self, model, num_envs, device, render_n_envs=16):
        """
        VecEnv which takes care of parallelization itself natively on GPU
        instead of vectorizing a single non-batch environment.
        Based on VecEnv interface by stable-baselines3.

        :param model:
        :param num_envs:
        :param device:
        :param render_n_envs:
        """
        self.device = device
        self.model = model
        self.num_envs = num_envs
        self.actions = None
        self.current_states = self.model.sample_new_states(num_envs)
        self.ep_stats = {
            'returns': torch.zeros((num_envs,), device=device),
            'lengths': torch.zeros((num_envs,), device=device),
        }
        self.render_n_envs = render_n_envs

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> TorchVecEnvStepReturn:
        with torch.no_grad():
            current_states = self.current_states

            if isinstance(self.actions, np.ndarray) or isinstance(self.actions, list):
                actions = torch.tensor(self.actions, device=self.device)
            else:
                # We are assuming that actions are on the correct device.
                actions = self.actions

            obses, rewards, dones, next_states = self.model.compute_step(
                current_states,
                actions)

        self.ep_stats['returns'] += rewards
        self.ep_stats['lengths'] += torch.ones_like(rewards)

        self.current_states = next_states

        n_dones = dones.sum()
        self.current_states[dones] = self.model.sample_new_states(n_dones)

        episode_returns = self.ep_stats['returns'][dones]
        episode_lengths = self.ep_stats['lengths'][dones]
        infos = (episode_returns, episode_lengths)

        self.ep_stats['returns'][dones] = 0
        self.ep_stats['lengths'][dones] = 0

        # Override observations for resetted environments after using them to
        # set "terminal_observation"
        obses[dones] = self.model.get_observations(self.current_states[dones])

        return obses.clone(), rewards.clone(), dones.clone(), infos

    def step(self, actions: np.ndarray):
        """
        Step the environments with the given action

        :param actions: the action
        :return: observation, reward, done, information
        """
        self.step_async(actions)
        return self.step_wait()

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """
        Environment-specific seeding is not used at the moment.
        In the underlying environment, random numbers are generated through
        calls to methods like torch.randn and can be seeded with
        `torch.manual_seed`.
        """
        raise NotImplementedError()

    def reset(self):
        self.current_states = self.model.sample_new_states(self.num_envs)
        obses = self.model.get_observations(self.current_states)
        return obses

    def get_images(self) -> Sequence[np.ndarray]:
        return [self.model.render(state) for state in self.current_states[:self.render_n_envs]]

    def render(self, mode: str) -> Optional[np.ndarray]:
        """
        Vendored from stable-baselines3
        """
        imgs = self.get_images()

        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs)
        if mode == "human":
            import cv2  # pytype:disable=import-error

            cv2.imshow("vecenv", bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported by VecEnvs")
