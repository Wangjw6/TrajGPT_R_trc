import pandas as pd
import math
import numpy as np
from shapely.geometry import Point, Polygon
import pickle
import os
import re
import pandas as pd


def extract_coordinates_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    pattern = re.compile(r',([0-9]+\.[0-9]+),([0-9]+\.[0-9]+)')
    matches = pattern.findall(content)
    coordinates = [(float(lat), float(lon)) for lon, lat in matches]
    return coordinates


def process_txt_files(file_path):
    column_names = ['ID', 'Timestamp', 'Longitude', 'Latitude']

    # Read the data into a pandas DataFrame
    df = pd.read_csv(file_path, header=None, names=column_names)
    return df
    all_coordinates = []
    coordinates = extract_coordinates_from_file(file_path)
    all_coordinates.extend(coordinates)
    return all_coordinates


def process_multiple_files(directory):
    all_coordinates = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            coordinates = extract_coordinates_from_file(file_path)
            all_coordinates.extend(coordinates)
    return all_coordinates


from scipy.interpolate import griddata

# Directory containing the text files
directory_path = r'E:\opendata\tdrive\taxi_log_2008_by_id\\'
# directory_path = r'./opendatahub/tdrive/taxi_log_2008_by_id/'
print("Reading files...")
# Process all files in the directory and get combined coordinates
beijing_lat_min, beijing_lat_max = 39.26, 41.03
beijing_lon_min, beijing_lon_max = 115.25, 117.30
from geopy.distance import great_circle, geodesic


def calculate_grid_id(lat, lon, lat_min, lon_min, grid_size):
    # Calculate distance in meters
    delta_lat = great_circle((lat_min, lon_min), (lat, lon_min)).meters
    delta_lon = great_circle((lat_min, lon_min), (lat_min, lon)).meters

    # Calculate grid indices
    lat_idx = int(delta_lat // grid_size)
    lon_idx = int(delta_lon // grid_size)
    return lat_idx, lon_idx


grid_size = 1000  # Example: 2000 meters
# get max lat lng index
lat_max_idx, lon_max_idx = calculate_grid_id(beijing_lat_max, beijing_lon_max, beijing_lat_min, beijing_lon_min,
                                             grid_size)
print("Gridizing")
max_gap = 0
distinct_grids = set()
minute_gap = 10
traj_num = 0
# for t in range(5592, 14000):
# action_dict = {"-1_-1": 0, "-1_0": 1, "-1_1": 2, "0_1": 3, "1_1": 4, "1_0": 5, "1_-1": 6, "0_-1": 7, "0_0": 8}
action_dict = {}
action_id = 2
action_dict["0_0"] = 0
combine_df = pd.DataFrame()
limit = 1
for t in range(0, 10400):
    file_path = directory_path + r'%d.txt' % t
    if not os.path.exists(file_path):
        continue
    # Process all files in the directory and get combined coordinates
    df = process_txt_files(file_path)
    df = df[(df['Latitude'] >= beijing_lat_min) & (df['Latitude'] <= beijing_lat_max) &
            (df['Longitude'] >= beijing_lon_min) & (df['Longitude'] <= beijing_lon_max)]
    df = df.drop_duplicates(subset='Timestamp', keep='first')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by='Timestamp')
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['day_of_month'] = df['Timestamp'].dt.day
    df['time_intervals'] = (df['Timestamp'].dt.hour * 60 + df['Timestamp'].dt.minute) // 10
    # continnue if no data
    if df.empty:
        continue
    # Process the coordinates and assign them to grids
    # e.g., 1 km grid size
    try:
        df[['lat_idx', 'lon_idx']] = df.apply(
            lambda row: calculate_grid_id(row['Latitude'], row['Longitude'], beijing_lat_min, beijing_lon_min,
                                          grid_size),
            axis=1,
            result_type='expand'
        )
    except:
        print(file_path)
        assert False

    # Create a unique grid ID based on lat_idx and lon_idx and the maximum longitude and latitude indices
    df['grid_id'] = df.apply(
        lambda row: row['lat_idx'] * (lat_max_idx + 1) + row['lon_idx'],
        axis=1
    )
    # df = df.drop_duplicates(subset=['lat_idx', 'lon_idx'], keep='first')
    distinct_grids.update(df['grid_id'].unique())

    df['time_diff'] = df['Timestamp'].diff().dt.total_seconds().div(60)  # Time difference in minutes
    df['trip_id'] = (df['time_diff'] > minute_gap).cumsum()
    # df['lat_diff'] = df.groupby('ID')['lat_idx'].diff().abs()
    # df['lon_diff'] = df.groupby('ID')['lon_idx'].diff().abs()
    # df["approx_speed"] = (df['lat_diff'] + df['lon_diff']) / df['time_diff'] * grid_size
    # df.loc[df.groupby('trip_id').head(1).index, ['lat_diff', 'lon_diff', 'approx_speed']] = df.loc[
    #     df.groupby('trip_id').head(1).index, ['lat_diff', 'lon_diff', 'approx_speed']].fillna(0)
    trip_id_list = [0]
    action_list = []
    grid_list = [df.iloc[0]['grid_id']]
    lat_list = [df.iloc[0]['lat_idx']]
    lon_list = [df.iloc[0]['lon_idx']]
    time_interval_list = [df.iloc[0]['time_intervals']]
    time_steps = [df.iloc[0]['Timestamp']]
    trip_id = 0
    # Loop through the DataFrame to update trip_id
    for i in range(1, len(df)):
        trip_id_list.append(trip_id)
        grid_list.append(df.iloc[i]['grid_id'])
        lat_list.append(df.iloc[i]['lat_idx'])
        lon_list.append(df.iloc[i]['lon_idx'])
        time_interval_list.append(df.iloc[i]['time_intervals'])
        time_steps.append(df.iloc[i]['Timestamp'])
        if abs(df.iloc[i]['lon_idx'] - df.iloc[i - 1]['lon_idx']) > limit or abs(df.iloc[i]['lat_idx'] - df.iloc[i - 1]['lat_idx']) > limit or np.abs(df.iloc[i]['time_diff']) > minute_gap:
            trip_id += 1
            # if len(action_list) > 0:
            #     if action_list[-1] == f"{limit+1}_{limit+1}":
            #         assert False
            action_list.append(f"{limit+1}_{limit+1}")
            continue
        relative_loc = f"{df.iloc[i]['lon_idx'] - df.iloc[i - 1]['lon_idx']}_{df.iloc[i]['lat_idx'] - df.iloc[i - 1]['lat_idx']}"
        action_list.append(relative_loc)
    if len(set(action_list)) <= 3:
        continue
    if action_list[-1] != f"{limit+1}_{limit+1}":
        assert len(action_list) == len(df)-1
        action_list.append(f"{limit+1}_{limit+1}")

    try:
        new_df = pd.DataFrame()
        new_df["ID"] = [t for _ in range(len(action_list))]
        new_df['grid_id'] = grid_list[:len(action_list)]
        new_df['lat_idx'] = lat_list[:len(action_list)]
        new_df['lon_idx'] = lon_list[:len(action_list)]
        new_df['action'] = action_list
        new_df['trip_id'] = trip_id_list[:len(action_list)]
        new_df['time_intervals'] = time_interval_list[:len(action_list)]
        new_df['Timestamp'] = time_steps[:len(action_list)]
    except:
        print(file_path)
        assert False
    # df = df[df['approx_speed'] < 1000]

    # Step 11: Drop rows where trip_id occurs less than 2 times
    trip_counts = new_df['trip_id'].value_counts()
    new_df = new_df[new_df['trip_id'].isin(trip_counts[trip_counts >= 4].index)]

    # Find the maximum gap
    # max_lat_gap = df.groupby('trip_id')['lat_diff'].max().max()
    # max_lon_gap = df.groupby('trip_id')['lon_diff'].max().max()
    # max_gap = max(max_gap, max_lat_gap, max_lon_gap)
    #
    # if max_gap > 10:
    #     print(file_path)

    # dump csv file
    if new_df.empty:
        continue
    trip_num = new_df['trip_id'].nunique()
    print(f"Files : {t}/10400 | number of grids {len(distinct_grids)} | total trips {traj_num}", end='\r')
    # df.to_csv(file_path.replace('.txt', '.csv'), index=False)
    combine_df = combine_df._append(new_df)
    traj_num += trip_num
combine_df.to_csv(f"combine.csv", index=False)
print("total number of grids", len(distinct_grids))
# print("max gap", max_gap)
print("total number of trajectories", traj_num)
