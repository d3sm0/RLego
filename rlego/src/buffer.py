import collections
import dataclasses
import numpy as np
import torch
from typing import List, Union

T = torch.Tensor


@dataclasses.dataclass
class Transition:
    state: T
    action: T
    reward: T
    next_state: T
    done: T
    info: Union[dict, List[dict]]

    def __iter__(self):
        return iter(dataclasses.asdict(self).values())

    @property
    def states(self):
        return self.state

    @property
    def actions(self):
        return self.action

    @property
    def rewards(self):
        return self.reward

    @property
    def next_states(self):
        return self.next_state

    @property
    def dones(self):
        return self.done

    @property
    def infos(self):
        return self.info


class Trajectory:
    def __init__(self):
        self.data: List[Transition] = []

    def __len__(self):
        return len(self.data)

    def append(self, transition: Transition):
        self.data.append(transition)

    def __iter__(self):
        return iter(self.data)

    def get_partial(self, start_idx, horizon) -> List[Transition]:
        horizon = min(self.__len__() - start_idx, horizon)
        return self.data[start_idx:start_idx + horizon]

    def sample(self, batch_size: int = 1, horizon: int = 1):
        start_idxs = torch.randint(self.__len__(), (batch_size,))
        # TODO we should be able to have something N X T but it seems hard for now we keep it like this
        return [self.get_partial(start_idx, horizon) for start_idx in start_idxs][0]

    def transpose(self) -> Transition:
        return _transpose(self.data)


def _transpose(data) -> Transition:
    if len(data) == 0:
        raise ValueError("Trajectory is empty")

    states, actions, rewards, next_states, dones, infos = list(zip(*self.data))
    states = torch.cat(states, 0)
    actions = magic_stack(actions)
    rewards = magic_stack(rewards)
    next_states = torch.cat(next_states, 0)
    dones = magic_stack(dones)
    infos2 = collections.defaultdict(list)
    for info in infos:
        for key, value in info.items():
            infos2[key].append(value)
    infos = {k: magic_stack(v) for k, v in infos2.items()}
    return Transition(states, actions, rewards, next_states, dones, infos)


def magic_stack(tensor_or_list: Union[torch.Tensor, list]) -> torch.Tensor:
    if isinstance(tensor_or_list[0], torch.Tensor):
        return torch.stack(tensor_or_list, 0)
    if isinstance(tensor_or_list[0], np.ndarray):
        return np.stack(tensor_or_list, 0)
    else:
        return torch.tensor(tensor_or_list)


class Buffer:
    def __init__(self, buffer_size: int):
        self._data = []
        self._buffer_size = buffer_size
        self._next_idx = 0

    def __len__(self):
        return len(self._data)

    def extend(self, trajectory: Trajectory):
        for transition in trajectory:
            self.append(transition)

    def append(self, transition: Transition):

        if self._next_idx >= len(self._data):
            self._data.append(transition)
        else:
            self._data[self._next_idx] = transition
        self._next_idx = (self._next_idx + 1) % self._buffer_size

    def sample(self, batch_size: int) -> Transition:
        idxes = torch.randint(len(self._data), (batch_size,))

        batch = [self._data[idx] for idx in idxes]
        return _transpose(batch)


def check_shape_transition(transition: Transition) -> bool:
    # TODO better error message
    assert len(transition.state.shape) == 2
    assert len(transition.reward.shape) == 1
    assert len(transition.next_state.shape) == 2
    assert len(transition.done.shape) == 1
    return True
