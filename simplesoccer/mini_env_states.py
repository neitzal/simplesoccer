from typing import Optional

import torch


class SoccerStates:
    """
    Holds a batch of SimpleSoccer states.
    """
    def __init__(self,
                 objects: torch.Tensor,
                 energies: torch.Tensor,
                 times: torch.Tensor,
                 opponent_states: Optional[torch.Tensor]):
        self.objects = objects
        self.energies = energies
        self.times = times
        self.opponent_states = opponent_states
        batch_shape = objects.shape[:-3]
        assert self.times.shape == batch_shape
        assert self.energies.shape[:-1] == batch_shape, f'{self.energies.shape}'
        if opponent_states is not None:
            self.opponent_states = opponent_states

    @property
    def batch_shape(self):
        return self.times.shape

    def __getitem__(self, item):
        if self.opponent_states is None:
            opponent_states = None
        else:
            opponent_states = tuple(s[item] for s in self.opponent_states)

        return SoccerStates(objects=self.objects[item],
                            energies=self.energies[item],
                            times=self.times[item],
                            opponent_states=opponent_states)

    def __setitem__(self, item, value):
        assert len(value.__dict__) == 4, f'Please account for other fields here: {value.__dict__}'

        self.objects[item] = value.objects
        self.energies[item] = value.energies
        self.times[item] = value.times
        if self.opponent_states is None:
            assert value.opponent_states is None
        else:
            assert isinstance(self.opponent_states, tuple)  # For now, only tuples are supported
            if value.opponent_states is None:
                v = 0
            else:
                v = value.opponent_states

            for s in self.opponent_states:
                s[item] = v

    @staticmethod
    def interpolate(states1, states2, alpha):
        if states1.opponent_states is None:
            assert states2.opponent_states is None
            opponent_states = None
        else:
            opponent_states = (1 - alpha) * states1.opponent_states + alpha*states2.oppponent_states
        return SoccerStates(
            objects=(1 - alpha)*states1.objects + alpha*states2.objects,
            energies=(1 - alpha) * states1.energies + alpha * states2.energies,
            times=(1 - alpha) * states1.times + alpha * states2.times,
            opponent_states=opponent_states
        )