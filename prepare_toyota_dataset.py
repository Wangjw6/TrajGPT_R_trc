"""
The input is the link connection dictionary (link_to_id.pkl, conn_dict.pkl)
and offline dataset for each car (#vehicle_id.pkl),
The output is RL trajectory data for each toyota car
"""
import os
import pickle
import numpy as np
import collections



def get_rl_data(scope="data_2m", usr='none'):  # data_2021101_1107_tokyo_core data_2m
    with open(f"./offline_data_hub/{scope}/conn_dictS.pkl", 'rb') as file:
        links = pickle.load(file)
    link_to_id = {}
    for k, v in links.items():
        if k not in link_to_id:
            link_to_id[k] = len(link_to_id)
        for c in v:
            if c not in link_to_id:
                link_to_id[c] = len(link_to_id)
    id_to_link = {v: k for k, v in link_to_id.items()}
    if usr == 'all':
        with open(f"./offline_data_hub/{scope}/rl_data_valid_usr.pkl", 'rb') as file:
            paths = pickle.load(file)
    else:
        if usr == 'none':
            with open(f"./offline_data_hub/{scope}/rl_data_valid.pkl", 'rb') as file:
                paths = pickle.load(file)
        else:
            try:
                with open(f"./offline_data_hub/{scope}/u{usr}_0_rl_data_valid.pkl", 'rb') as file:
                    paths = pickle.load(file)
            except:
                with open(f"./offline_data_hub/{scope}/u{usr}_1_rl_data_valid.pkl", 'rb') as file:
                    paths = pickle.load(file)
    # clean the traj to remove two conseceutive links in a trajectory
    clean_paths = []
    action = []
    traj_len = []
    link_in_data = []
    time_tag = []
    action_counts = {}
    usr = 0
    for i in range(len(paths)):
        # for k, path in paths.items():
        path = paths[i]
        observations = np.array(path['observations'])
        if np.max(observations) >= 262143:
            # print("Link out of range")
            continue
        if observations.shape[0] == 0:
            continue
        selected_idx = [0]
        exist = [observations[0][2]]
        is_duplicate = False
        for j in range(1, len(observations)):
            if observations[j][2] not in exist:
                exist.append(observations[j][2])
            else:
                is_duplicate = True
                break
            link_in_data.append(observations[j][2])

            if observations[j][2] != observations[j - 1][2]:
                selected_idx.append(j)
        if is_duplicate:
            continue
        if np.max(observations[:, 2]) >= 2 ** 19:
            continue
        new_observations = observations[selected_idx].tolist()

        begin_time = new_observations[0][4]
        time_tag.append(begin_time)
        begin_speed = new_observations[0][3]
        for t in range(len(new_observations)):
            new_observations[t] = np.concatenate([new_observations[t], [begin_time]])
            new_observations[t][4] = t
            new_observations[t][3] = begin_speed
        link_sequence = [obs[2] for obs in new_observations]
        if len(link_sequence) != len(set(link_sequence)):
            print("Duplicate links in trajectory")
            continue
        if new_observations[0][0] != new_observations[0][2]:
            assert False
        if new_observations[-1][1] != new_observations[-1][2]:
            assert False
        path['observations'] = np.array(new_observations)
        usr = max(usr, path['observations'][0][5])
        path['actions'] = np.array(path['actions'])[selected_idx]
        assert np.where(path['actions'] == 0)[0].shape[0] == 1
        action += path['actions'].tolist()
        path['rewards'] = np.array(path['rewards'])[selected_idx]
        path['terminals'] = np.array(path['terminals'])[selected_idx]
        path['next_observations'] = path['observations'][1:]
        if len(path['observations']) < 2:
            continue
        clean_paths.append(path)
        traj_len.append(len(new_observations))
    print(f'Min of link: {min(link_in_data)}')
    print(f'Max of link: {max(link_in_data)}')
    print(f'Len: {len(list(set(link_in_data)))}')
    print(f'Max of actions: {max(action)}')
    print(f'Min of actions: {min(action)}')
    print(f'Min of time: {min(time_tag)}')
    print(f'Len of time: {max(time_tag)}')
    print(f'Max of usr id: {usr}')

    for a in action:
        if a not in action_counts:
            action_counts[a] = 1
        else:
            action_counts[a] += 1
    for k, v in action_counts.items():
        print(f"Action {k}: {v}")
    print("Trajectory length: ", max(traj_len))
    # print the elements set that comprise 80% of the data
    # most_frequent(clean_paths)
    # print the length that occurs most frequently
    print("Most frequent trajectory length: ", max(set(traj_len), key=traj_len.count))
    print("Time tag: ", min(time_tag), max(time_tag))
    print("Number of trajectories: ", len(clean_paths))
    # permute the trajectories
    np.random.shuffle(clean_paths)
    return clean_paths[:int(0.8 * len(clean_paths))], clean_paths[int(0.8 * len(clean_paths)):], link_to_id





if __name__ == '__main__':
    # prepare_multi()
    get_rl_data()
