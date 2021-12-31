import dataclasses

import torch


@dataclasses.dataclass
class Transition:
    state: torch.tensor
    action: torch.tensor
    reward: torch.tensor
    next_state: torch.tensor
    done: torch.tensor
    info: dict

    def __iter__(self):
        for attr in ["state", "action", "reward", "next_state", "done", "info"]:
            yield getattr(self, attr)


class Trajectory:
    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def append(self, transition: Transition):
        self.data.append(transition)

    def __iter__(self):
        return iter(self.data)

    def get_partial(self, start_idx, horizon):
        horizon = min(self.__len__() - start_idx, horizon)
        return self.data[start_idx:start_idx + horizon]

    def sample(self, batch_size: int = 1, horizon: int = 1):
        start_idxs = torch.randint(self.__len__() - 1, (batch_size,))
        # TODO we should be able to have something N X T but it seems hard for now we keep it like this
        return [self.get_partial(start_idx, horizon) for start_idx in start_idxs][0]

    def get_trajectory(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, List[dict]]:
        r_t, a_t, s_t, s_tp1, infos = list(zip(*self.data))
        r_t = torch.tensor(r_t)
        a_t = torch.stack(a_t, 0)
        s_t = torch.stack(s_t, 0)
        s_tp1 = torch.stack(s_tp1, 0)
        return r_t, a_t, s_t, s_tp1, infos


class Buffer:
    def __init__(self, buffer_size):
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

    def sample(self, batch_size: int):
        idxes = torch.randint(len(self._data) - 1, (batch_size,))

        batch = [self._data[idx] for idx in idxes]
        batch = list(map(lambda x: torch.stack(x), list(zip(*batch))))

        return Transition(*batch)
