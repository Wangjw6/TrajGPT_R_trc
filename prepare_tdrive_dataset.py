"""
The input is the one-day trajectory data from dump_data.py
Output will be link connection dictionary (link_to_id.pkl, conn_dict.pkl)
and offline dataset for each car (#vehicle_id.pkl), which will be transformed by (prepare_toyota_dataset.py) for training data in the generation model
"""
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from collections import Counter
from time import time

pd.options.mode.chained_assignment = None  # default='warn'
OD = {}
link_to_id = {}
conn_dict = {}
downstream_links = {}
tokens = []
sentences = []


# Check if the file path is provided as a command-line argument


def step(obs, action, id_to_link, link_to_id):
    action = np.array(action).astype(int)
    current_link = obs[0][0]
    try:
        connect_links = find_connect(id_to_link[current_link])
    except Exception as e:
        print(e)
        return obs, 0., False
    # print(connect_links)
    r = 0.
    flag = False
    if len(connect_links) > action:
        next_link = connect_links[action[0]]
        r = 0.
    else:
        next_link = id_to_link[current_link]

    try:
        next_link = link_to_id[next_link]
        # print('next', next_link)
    except:
        flag = True

    next_obs = [next_link, obs[0][1], obs[0][2], obs[0][3], obs[0][4]]
    return next_obs, r, flag


def find_connect(link, lat=None, lng=None, hint=None):
    link = int(link)

    return connect


import math


def transform_dt(x):
    try:
        datetime_obj = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    except:
        datetime_obj = datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
    return datetime_obj


import ast


def format_dataset(car, dataset, action_dict):
    # Filter the dataframe for the specific car
    df_ = df[df["ID"] == car]
    fake_trip_id = 0
    # df_.reset_index(drop=True, inplace=True)
    # for i in range(0, df_.shape[0]-1):
    #     if df_.iloc[i, -2] == "2_2":
    #         df_.iloc[i+1, -2] = max(df_.iloc[i, -2]+1, df_.iloc[i+1, -2])
    #     else:
    #         df_.iloc[i+1, -2] = df_.iloc[i, -2]
    # rows_to_delete = []
    # # Step 1: Remove rows with large jumps in latitude and longitude
    # flag = True
    for i in range(1, df_.shape[0]):
        if abs(df_.iloc[i, 6] - df_.iloc[i - 1, 6]) > 10 or df_.iloc[i - 1, 5] == "2_2":
            fake_trip_id += 1
            df_.iloc[i, 5] += fake_trip_id
        else:
            df_.iloc[i, 5] += fake_trip_id

    mask = (df_['action'] == "2_2") & (df_['action'].shift() == "2_2")
    # Drop rows that are repetitive (only keep the first one)
    df_filtered = df_[~mask]
    grouped = df_filtered.groupby('trip_id')

    sorted_grouped = grouped.apply(lambda x: x.sort_values(['trip_id', "time_intervals"])).reset_index(drop=True)

    # Then, aggregate the sorted data by concatenating the values
    this_car = sorted_grouped.groupby('trip_id').agg({
        'lat_idx': lambda x: '_'.join(map(str, x)),
        'lon_idx': lambda x: '_'.join(map(str, x)),
        'grid_id': lambda x: '_'.join(map(str, x)),
        'time_intervals': lambda x: '_'.join(map(str, x)),
        'action': lambda x: '#'.join(x)  # Concatenate strings in 'action' with '#'
    }).reset_index()

    log = ""
    for i in range(this_car.shape[0]):
        episode_link = [int(s) for s in (this_car["grid_id"].iloc[i]).split("_")]
        episode_link = np.array(episode_link)
        episode_lat = [int(s) for s in (this_car["lat_idx"].iloc[i]).split("_")]
        episode_lon = [int(s) for s in (this_car["lon_idx"].iloc[i]).split("_")]
        episode_lat = np.array(episode_lat)
        episode_lon = np.array(episode_lon)
        episode_intervals = [int(s) for s in (this_car["time_intervals"].iloc[i]).split("_")]
        episode_actions = [str(s) for s in (this_car["action"].iloc[i]).split("#")]
        if len(list(set(episode_actions))) <= 3:
            continue
        episode_intervals = np.array(episode_intervals)
        observation = np.concatenate(
            [episode_link.reshape(-1, 1), episode_lat.reshape(-1, 1), episode_lon.reshape(-1, 1),
             episode_intervals.reshape(-1, 1)], axis=1)
        reward = np.zeros(len(episode_link))
        # reward[-1] = 1.
        action = []
        conn_bad = 0
        while True:
            record = []
            for j in range(len(episode_link) - 1):
                lat_shift = int(episode_lat[j + 1]) - int(episode_lat[j])
                lon_shift = int(episode_lon[j + 1]) - int(episode_lon[j])
                a = f"{lon_shift}_{lat_shift}"
                if a != episode_actions[j]:
                    assert False
                if a not in action_dict:
                    assert False
                action.append(action_dict[a])

            if len(action) == len(episode_link) - 1:
                action.append(10)
                break

        dones = np.zeros(len(episode_link))
        dones[-1] = 1
        episode_actions = action
        episode_rewards = reward.tolist()
        episode_dones = dones.tolist()
        episode_observations = np.zeros([len(episode_link), 5])
        # episode_observations[:, 0] = episode_link[0]
        # episode_observations[:, 1] = episode_link[-1]
        # episode_observations[:, 2:] = observation
        episode_observations = observation.tolist()
        assert len(episode_observations) == len(episode_actions) == len(episode_rewards) == len(episode_dones)
        dataset['observations'].extend(episode_observations)
        dataset['actions'].extend(episode_actions)
        dataset['rewards'].extend(episode_rewards)
        dataset['terminals'].extend(episode_dones)
        dataset['timeouts'].extend(episode_dones)

    if len(log) > 1:
        with open(f"./error_log.txt", 'a') as file:
            file.write(log)
    return dataset, None, None


import multiprocessing

if __name__ == "__main__":
    datasets = {}
    try:
        df = pd.read_csv("./offline_data_hub/opendata/combine.csv")
    except:
        df = pd.read_csv("./combine.csv")
        unique_grid = df['grid_id'].unique().tolist()
        print("unique_grid", len(unique_grid))
        # grid_loc_dict = {row['grid_id']: f"{row['lat_idx']}_{row['lon_idx']}" for _, row in df.iterrows()}
        # # reverse the grid_dict
        # loc_grid_dict = {v: k for k, v in grid_loc_dict.items()}
        # # dump the grid_dict
        # with open(f"./tdrive_grid_loc_dict.pkl", 'wb') as file:
        #     # Pickle the 'data' dictionary using the highest protocol available.
        #     pickle.dump(grid_loc_dict, file, pickle.HIGHEST_PROTOCOL)
        # with open(f"./tdrive_loc_grid_dict.pkl", 'wb') as file:
        #     # Pickle the 'data' dictionary using the highest protocol available.
        #     pickle.dump(loc_grid_dict, file, pickle.HIGHEST_PROTOCOL)
    # reorganize the action
    raw_grids = df['grid_id'].unique().tolist()
    raw_action = df['action'].unique().tolist()
    lon_shift_list = []
    lat_shift_list = []

    action_dict = {}
    for a in raw_action:
        lon_shift = int(a.split('_')[0])
        lat_shift = int(a.split('_')[1])
        # if lon_shift == 11 and lat_shift == 11:
        #     action_dict[a] = 121
        #     continue
        action_dict[a] = -1
    # Sort the dictionary first by the first element in 'a_b' and then by the second element
    sorted_by_first_element = dict(
        sorted(action_dict.items(), key=lambda x: (int(x[0].split('_')[0]), int(x[0].split('_')[1]))))
    a = 0
    for k, v in sorted_by_first_element.items():
        sorted_by_first_element[k] = a
        a += 1
    # dump action dict
    action_dict = sorted_by_first_element
    with open(f"./tdrive_action_dict.pkl", 'wb') as file:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(action_dict, file, pickle.HIGHEST_PROTOCOL)
    # dump inverse action dict
    with open(f"./tdrive_inverse_action_dict.pkl", 'wb') as file:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump({v: k for k, v in action_dict.items()}, file, pickle.HIGHEST_PROTOCOL)

    # print the minimum value in action_dict
    assert min(action_dict.values()) >= 0
    print(max(action_dict.values()))

    unique_cars = list(set(df['ID'].unique().tolist()))
    print("unique_cars", len(unique_cars))

    i = 1
    episode_num = 0
    action_space = []
    for car in unique_cars:
        dataset = {}
        dataset['observations'] = []
        dataset['actions'] = []
        dataset['rewards'] = []
        dataset['terminals'] = []
        dataset['timeouts'] = []
        dataset, link_to_id_, links = format_dataset(car, dataset, action_dict)
        # print(f"car {i} | {len(unique_cars)} has been processed")

        if len(dataset['rewards']) == 0:
            continue
        datasets[car] = dataset
        episode_num += sum(dataset['terminals'])
        i += 1
        print(f"Data Formulating Progress: {i}/{len(unique_cars)} | Collected episode_num: {episode_num} ", end='\r')

    print("episode_num", episode_num)
    # with open(f"./offline_data_hub/data_2021101_1107_tokyo_core/sentences.pkl", 'wb') as file:
    #     # Pickle the 'data' dictionary using the highest protocol available.
    #     pickle.dump(sentences, file, pickle.HIGHEST_PROTOCOL)
    with open(f"./tdrive_vehiclesS.pkl", 'wb') as file:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(datasets, file, pickle.HIGHEST_PROTOCOL)
