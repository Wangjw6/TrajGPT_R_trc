from typing import Any, Dict, IO, List, Tuple

import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import os

def process_data(trajectories, seq_len=256):
    states = []
    traj_lens = []
    returns = []

    for path in trajectories:
        # try:
        #     state = np.concatenate([path['observations'].reshape(-1, 4)[:, :], path['actions'].reshape(-1, 1)], axis=1)
        # except:
        state = np.concatenate([path['observations'].reshape(-1, 7)[:, :], path['actions'].reshape(-1, 1)], axis=1)
        states.append(state)
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)
    # patch for lstm
    max_len = max(traj_lens)
    max_len = max(max_len, seq_len)
    for i in range(len(states)):
        states[i] = np.concatenate([states[i], np.zeros((max_len - len(states[i]), states[i].shape[1]))])
    states = np.array(states)
    states = torch.from_numpy(states).float()
    traj_lens = torch.from_numpy(traj_lens).long()
    return states, traj_lens, returns

class ExpertDataset(Dataset):
    """Dataset for expert trajectories.

    Assumes expert dataset is a dict with keys {states, actions, rewards, lengths} with values containing a list of
    expert attributes of given shapes below. Each trajectory can be of different length.

    Expert rewards are not required but can be useful for evaluation.

        shapes:
            expert["observations"]  =  [num_experts, traj_length, state_space]
            expert["actions"] =  [num_experts, traj_length, action_space]
            expert["rewards"] =  [num_experts, traj_length]
            expert["lengths"] =  [num_experts]
    """

    def __init__(self,
                 all_trajectories,
                 num_trajectories: int = 4,
                 subsample_frequency: int = 20,
                 seed: int = 0):
        """Subsamples an expert dataset from saved expert trajectories.

        Args:
            expert_location:          Location of saved expert trajectories.
            num_trajectories:         Number of expert trajectories to sample (randomized).
            subsample_frequency:      Subsamples each trajectory at specified frequency of steps.
            deterministic:            If true, sample determinstic expert trajectories.
        """
        # all_trajectories = load_trajectories(expert_location, num_trajectories, seed)
        self.trajectories = {}

        # Randomize start index of each trajectory for subsampling
        # start_idx = torch.randint(0, subsample_frequency, size=(num_trajectories,)).long()

        # Subsample expert trajectories with every `subsample_frequency` step.
        self.length = len(all_trajectories)
        for i in range(len(all_trajectories)):
            trajecotry = all_trajectories[i]
            states, traj_lens, returns = process_data([trajecotry], 256)
            for k, v in trajecotry.items():
                data = v
                if k != "lengths":
                    # samples = []
                    # samples.append(data[0::subsample_frequency])
                    if k not in self.trajectories:
                        self.trajectories[k] = []
                    self.trajectories[k].append(data[0::subsample_frequency])
            if "lengths" not in self.trajectories:
                self.trajectories["lengths"] = []
                self.trajectories["full_traj"] = []
            self.trajectories["lengths"].append(len(v))
            self.trajectories["full_traj"].append([states, traj_lens, returns])

        # for k, v in all_trajectories.items():
        #     data = v
        #     if k != "lengths":
        #         samples = []
        #         for i in range(num_trajectories):
        #             samples.append(data[i][0::subsample_frequency])
        #         self.trajectories[k] = samples
        #     else:
        #         # Adjust the length of trajectory after subsampling
        #         self.trajectories[k] = np.array(data) // subsample_frequency

        self.i2traj_idx = {}
        # self.length = self.trajectories["lengths"].sum().item()
        del all_trajectories  # Not needed anymore
        traj_idx = 0
        i = 0

        # Convert flattened index i to trajectory indx and offset within trajectory
        self.get_idx = []

        for _j in range(int(self.length)):
            while self.trajectories["lengths"][traj_idx] <= i:
                i -= self.trajectories["lengths"][traj_idx]
                traj_idx += 1

            self.get_idx.append((traj_idx, i))
            i += 1

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def __getitem__(self, i):
        traj_idx, i = self.get_idx[i]
        try:
            states = self.trajectories["observations"][traj_idx][i]
            next_states = self.trajectories["next_observations"][traj_idx][i]
        except:
            print(traj_idx, i, len(self.trajectories["observations"][traj_idx]), len(self.trajectories["next_observations"][traj_idx]))


        return (states,
                next_states,
                self.trajectories["actions"][traj_idx][i],
                self.trajectories["rewards"][traj_idx][i],
                self.trajectories["rewards"][traj_idx][i],
                self.trajectories["full_traj"][traj_idx])


def load_trajectories(expert_location: str,
                      num_trajectories: int = 10,
                      seed: int = 0) -> Dict[str, Any]:
    """Load expert trajectories

    Args:
        expert_location:          Location of saved expert trajectories.
        num_trajectories:         Number of expert trajectories to sample (randomized).
        deterministic:            If true, random behavior is switched off.

    Returns:
        Dict containing keys {"observations", "lengths"} and optionally {"actions", "rewards"} with values
        containing corresponding expert data attributes.
    """
    if os.path.isfile(expert_location):
        # Load data from single file.
        with open(expert_location, 'rb') as f:
            trajs = read_file(expert_location, f)

        rng = np.random.RandomState(seed)
        # Sample random `num_trajectories` experts.
        perm = np.arange(len(trajs["observations"]))
        perm = rng.permutation(perm)

        idx = perm[:num_trajectories]
        for k, v in trajs.items():
            # if not torch.is_tensor(v):
            #     v = np.array(v)  # convert to numpy array
            trajs[k] = [v[i] for i in idx]

    else:
        raise ValueError(f"{expert_location} is not a valid path")
    return trajs


def read_file(path: str, file_handle: IO[Any]) -> Dict[str, Any]:
    """Read file from the input path. Assumes the file stores dictionary data.

    Args:
        path:               Local or S3 file path.
        file_handle:        File handle for file.

    Returns:
        The dictionary representation of the file.
    """
    if path.endswith("pt"):
        data = torch.load(file_handle)
    elif path.endswith("pkl"):
        data = pickle.load(file_handle)
    elif path.endswith("npy"):
        data = np.load(file_handle, allow_pickle=True)
        if data.ndim == 0:
            data = data.item()
    else:
        raise NotImplementedError
    return data