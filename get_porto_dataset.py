import os
import pickle
import numpy as np
import collections


def get_rl_data(scope="opendata", usr="None"):
    # with open(f"./offline_data_hub/{scope}/rl_gldata_valid.pkl", 'rb') as file:
    #     paths = pickle.load(file)
    if usr == 'all':
        try:
            with open(f"./offline_data_hub/{scope}/porto/rl_gldata_valid_usr.pkl", 'rb') as file:
                paths = pickle.load(file)
        except EOFError:
            print("Error: File is empty or corrupted.")
        except pickle.UnpicklingError:
            print("Error: File is not a valid pickle format.")
    else:
        if usr == 'none':
            with open(f"./offline_data_hub/{scope}/porto/rl_data_valid.pkl", 'rb') as file:
                paths = pickle.load(file)
        else:
            try:
                with open(f"./offline_data_hub/{scope}/porto/u{usr}_0_rl_data_valid.pkl", 'rb') as file:
                    paths = pickle.load(file)
            except:
                with open(f"./offline_data_hub/{scope}/porto/u{usr}_1_rl_data_valid.pkl", 'rb') as file:
                    paths = pickle.load(file)
    # clean the traj to remove two conseceutive links in a trajectory
    clean_paths = []
    action = []
    traj_len = []
    link_in_data = []
    lat_tokens = []
    lon_tokens = []
    grid_tokens = []
    time_tag = []
    action_counts = {}
    grid_to_id = {}
    max_len = 0
    usr_id = 0
    for path in paths:
        # for k, path in paths.items():
        observations = np.array(path['observations'])  # grid, lat, lon, interval
        if observations.shape[0] == 0:
            continue
        selected_idx = [0]
        begin_time = observations[0][3]
        observations[:, 3] = begin_time
        for j in range(1, len(observations)):
            if observations[j][2] != observations[j - 1][2] or observations[j][1] != observations[j - 1][1]:
                selected_idx.append(j)

        new_observations = observations.tolist()  # [selected_idx].tolist()
        grid_tokens += observations[:, 0].tolist()
        lat_tokens += observations[:, 1].tolist()
        lon_tokens += observations[:, 2].tolist()

        time_tag.append(begin_time)
        # add od token
        origin = new_observations[0][0]
        destination = new_observations[-1][0]
        usr_id = max(usr_id, new_observations[-1][4])
        new_observations = [
            [new_observations[i][0], new_observations[i][1], new_observations[i][2], begin_time, origin, destination,
             new_observations[i][4]] for i in range(len(new_observations))]
        path['observations'] = np.array(new_observations)
        path['actions'] = np.array(path['actions'])  # [selected_idx]
        if np.where(path['actions'] == 9)[0].shape[0] != 1 or path['actions'][-1] != 9:
            assert False
        link_in_this_traj = path['observations'][:, 0].reshape(-1).tolist()
        if len(set(link_in_this_traj)) / len(link_in_this_traj) <= 0.2:
            continue
        link_in_data += link_in_this_traj
        action += path['actions'].tolist()
        path['rewards'] = np.array(path['rewards'])  # [selected_idx]
        path['rewards'][:-1] = 0
        path['terminals'] = np.array(path['terminals'])  # [selected_idx]
        path['next_observations'] = path['observations'][1:]
        if len(set(path['actions'].tolist())) < 2:
            continue
        max_len = max(max_len, len(new_observations))
        if len(path['observations']) > 256:
            continue

        clean_paths.append(path)
        traj_len.append(len(new_observations))
    temp = {}
    k = 0
    for i in range(len(grid_tokens)):
        lat_lng = (lat_tokens[i], lon_tokens[i])
        if grid_tokens[i] not in temp:
            temp[grid_tokens[i]] = k
            assert lat_lng not in grid_to_id
            grid_to_id[lat_lng] = k
            k += 1

    for i in range(len(clean_paths)):
        clean_paths[i]['observations'][:, 0] = [temp[l] for l in clean_paths[i]['observations'][:, 0]]
        clean_paths[i]['observations'][:, 5] = [temp[l] for l in clean_paths[i]['observations'][:, 5]]
        clean_paths[i]['observations'][:, 4] = [temp[l] for l in clean_paths[i]['observations'][:, 4]]

    print(f'Min of link: {min(link_in_data)}')
    print(f'Max of link: {max(link_in_data)}')
    print(f'Min of lat: {min(lat_tokens)}')
    print(f'Max of lat: {max(lat_tokens)}')
    print(f'Min of lon: {min(lon_tokens)}')
    print(f'Max of lon: {max(lon_tokens)}')
    print(f'Len: {len(list(set(link_in_data)))}')
    print(f'Max of actions: {max(action)}')
    print(f'Min of actions: {min(action)}')
    print(f'Len of actions: {len(list(set(action)))}')
    print(f'Min of time: {min(time_tag)}')
    print(f'max of time: {max(time_tag)}')
    print(f'Len of grid: {len(grid_to_id)}')
    print(f'max user: {usr_id}')

    print("Trajectory length: ", max(traj_len))
    print("Most frequent trajectory length: ", max(set(traj_len), key=traj_len.count))
    print("Time tag: ", min(time_tag), max(time_tag))
    print("Number of trajectories: ", len(clean_paths))
    # permute the trajectories
    np.random.shuffle(clean_paths)
    return clean_paths[:int(0.8 * len(clean_paths))], clean_paths[int(0.8 * len(clean_paths)):], grid_to_id


def get_gail_data(scope="opendata", usr=None):
    if usr == 'all':
        with open(f"./offline_data_hub/{scope}/porto/rl_gldata_valid_usr.pkl", 'rb') as file:
            paths = pickle.load(file)
    else:
        if usr == 'none':
            with open(f"./offline_data_hub/{scope}/porto/rl_gldata_valid_usr.pkl", 'rb') as file:
                paths = pickle.load(file)
        else:
            assert False

    clean_paths = []
    action = []
    traj_len = []
    link_in_data = []
    lat_tokens = []
    lon_tokens = []
    grid_tokens = []
    time_tag = []
    action_counts = {}
    grid_to_id = {}
    usr_id = 0
    for path in paths:
        # for k, path in paths.items():
        observations = np.array(path['observations'])  # grid, lat, lon, interval
        if observations.shape[0] == 0:
            continue
        selected_idx = [0]
        begin_time = observations[0][3]
        observations[:, 3] = begin_time
        for j in range(1, len(observations)):
            if observations[j][2] != observations[j - 1][2] or observations[j][1] != observations[j - 1][1]:
                selected_idx.append(j)

        new_observations = observations.tolist()  # [selected_idx].tolist()
        grid_tokens += observations[:, 0].tolist()
        lat_tokens += observations[:, 1].tolist()
        lon_tokens += observations[:, 2].tolist()

        time_tag.append(begin_time)
        # add od token
        origin = new_observations[0][0]
        destination = new_observations[-1][0]
        usr_id = max(usr_id, new_observations[-1][4])
        new_observations = [[new_observations[i][0], new_observations[i][1], new_observations[i][2], begin_time] for i
                            in range(len(new_observations))]  # grid, lng, lat,  depart
        path['observations'] = np.array(new_observations)
        path['actions'] = np.array(path['actions'])  # [selected_idx]
        if np.where(path['actions'] == 9)[0].shape[0] != 1 or path['actions'][-1] != 9:
            assert False
        link_in_this_traj = path['observations'][:, 0].reshape(-1).tolist()
        if len(set(link_in_this_traj)) / len(link_in_this_traj) <= 0.2:
            continue
        link_in_data += link_in_this_traj
        action += path['actions'].tolist()
        path['rewards'] = np.array(path['rewards'])  # [selected_idx]
        path['rewards'][:-1] = 0
        path['terminals'] = np.array(path['terminals'])  # [selected_idx]
        path['next_observations'] = path['observations'][1:]
        if len(set(path['actions'].tolist())) < 2 or len(path['observations']) > 256:
            continue
        if path['observations'][0, 0] == path['observations'][-1, 0]:
            continue
        clean_paths.append(path)
        traj_len.append(len(new_observations))
    temp = {}
    k = 0
    for i in range(len(grid_tokens)):
        lat_lng = (lat_tokens[i], lon_tokens[i])
        if grid_tokens[i] not in temp:
            temp[grid_tokens[i]] = k
            assert lat_lng not in grid_to_id
            grid_to_id[lat_lng] = k
            k += 1
    # update grid tokens in the observations
    for i in range(len(clean_paths)):
        clean_paths[i]['observations'][:, 0] = [temp[l] for l in clean_paths[i]['observations'][:, 0]]

    print(f'Min of link: {min(link_in_data)}')
    print(f'Max of link: {max(link_in_data)}')
    print(f'Min of lat: {min(lat_tokens)}')
    print(f'Max of lat: {max(lat_tokens)}')
    print(f'Min of lon: {min(lon_tokens)}')
    print(f'Max of lon: {max(lon_tokens)}')
    print(f'Len: {len(list(set(link_in_data)))}')
    print(f'Max of actions: {max(action)}')
    print(f'Min of actions: {min(action)}')
    print(f'Len of actions: {len(list(set(action)))}')
    print(f'Min of time: {min(time_tag)}')
    print(f'max of time: {max(time_tag)}')
    print(f'Len of grid: {len(grid_to_id)}')
    print(f'max user: {usr_id}')

    print("Trajectory length: ", max(traj_len))
    print("Most frequent trajectory length: ", max(set(traj_len), key=traj_len.count))
    print("Time tag: ", min(time_tag), max(time_tag))
    print("Number of trajectories: ", len(clean_paths))
    # permute the trajectories
    np.random.shuffle(clean_paths)
    return clean_paths[:int(0.8 * len(clean_paths))], clean_paths[int(0.8 * len(clean_paths)):], grid_to_id


if __name__ == '__main__':
    # prepare_multi()
    get_rl_data()
