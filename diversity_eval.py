import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import math
import editdistance

Inh_df = pd.read_csv('E:/toyota/network/small.csv',  encoding='shift_jis')
# Define boundaries and grid step sizes
max_lat = 35.91521267361111
min_lat = 35.5
max_lon = 139.99993706597223
min_lon = 139.0001801215278

# 1 km in latitude and longitude step size
lat_step = 4 / 111  # ~0.009 degrees
central_lat = (max_lat + min_lat) / 2
lon_step = 4 / (111 * math.cos(math.radians(central_lat)))  # Adjust for latitude
import pickle

with open(f"E:\\toyota\\offline_data_hub\\data_2m\\conn_dictS.pkl", 'rb') as file:
    links = pickle.load(file)
link_to_id = {}
for k, v in links.items():
    if k not in link_to_id:
        link_to_id[k] = len(link_to_id)
    for c in v:
        if c not in link_to_id:
            link_to_id[c] = len(link_to_id)
id_to_link = {v: k for k, v in link_to_id.items()}

# Function to find grid index
def find_grid_index(lat, lon, min_lat, min_lon, lat_step, lon_step):
    if lat < min_lat or lat > max_lat or lon < min_lon or lon > max_lon:
        return None  # Out of bounds

    # Calculate row and column indices
    row = int((lat - min_lat) / lat_step)
    col = int((lon - min_lon) / lon_step)

    return row, col

# Configure global settings for Nature-style figures
rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.5,
    'legend.fontsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
})
prefixs = ['My_Transformer_od_ft#usr_','My_Transformer_od_fttdriveusr_','My_Transformer_od_ftgeolifeusr_',]
models = ['ad', 'dpo', 'jd', 'rlhfw0', 'pretrained']
models = [ 'dpo',   'rlhfw0', 'pretrained']
other_models_full = {'geo': ['TrajGAIL_gen_geo', 'IQL_gen_geo', 'DiffTraj_gen_geo'],
                     'drive': ['TrajVAE_gen_drive','TrajGAIL_gen_drive', 'IQL_gen_drive', 'DiffTraj_gen_drive'],
                     'toyota': ["TrajVAE_gen_toyota", 'TrajGAIL_gen_toyota', 'IQL_gen_toyota', 'DiffTraj_gen_toyota']}
full_model_name = {"ad": r"$\alpha$-DPO", "dpo": "DPO", "jd": "Jsd-DPO", "rlhfw0": "RLHF",
                   "Pretrained": "Pretrained",
                   "TrajGAIL_gen_geo": "TrajGAIL", "IQL_gen_geo": "IQL", "DiffTraj_gen_geo": "D3PM",
                   "TrajGAIL_gen_drive": "TrajGAIL", "IQL_gen_drive": "IQL", "DiffTraj_gen_drive": "D3PM", "TrajVAE_gen_drive": "TrajVAE",
                   "TrajGAIL_gen_toyota": "TrajGAIL", "IQL_gen_toyota": "IQL", "DiffTraj_gen_toyota": "D3PM", "TrajVAE_gen_toyota": "TrajVAE",}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e41a1c', '#377eb8', '#4daf4a',
          '#984ea3']  # Muted Nature palette
line_styles = ['-', '--', '-.', ':', '-']  # Different line styles for distinction
markers = ['o', 's', 'D', '^', 'v']  # Distinct markers for each model

trajectories_set = {}


def hierarchical_clustering(trajectory_strings):
    # Bag of Links representation
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(trajectory_strings).toarray()
    linkage_matrix = linkage(X, method="ward")  # 'ward', 'single', 'complete', etc.

    # Plot dendrogram
    plt.figure(figsize=(10, 6))
    dendrogram(
        linkage_matrix,
        labels=[f"Trajectory {i + 1}" for i in range(len(trajectory_strings))],
        leaf_rotation=90,
        leaf_font_size=12,
        color_threshold=5.0,  # Adjust to highlight clusters
    )
    plt.title("Hierarchical Clustering of Trajectories")
    plt.xlabel("Trajectories")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

    # Extract flat clusters (optional)
    threshold = 5.0  # Set a threshold to define clusters
    cluster_labels = fcluster(linkage_matrix, t=threshold, criterion="distance")
    print(len(set(cluster_labels)))
    # Display cluster assignments
    # for i, label in enumerate(cluster_labels):
    #     print(f"Trajectory {i + 1}: Cluster {label}")


def selfbleu_trajectories(trajectories_set, n_clusters=5):
    for k, trajectories in trajectories_set.items():
        # Set Nature journal-style aesthetics
        sns.set_context("paper", font_scale=1.5)
        sns.set_style("whitegrid")
        # tolist
        if k == 'truth':
            od_trajectories = {}
            for real_trajectory in trajectories:
                origin = int(real_trajectory[0])
                destination = int(real_trajectory[-1])
                if f"{origin}_{destination}" not in od_trajectories:
                    od_trajectories[f"{origin}_{destination}"] = []
                od_trajectories[f"{origin}_{destination}"].append(real_trajectory)
        else:
            idx = 0
            gen_trajectories = []
            for traj in trajectories:
                gen_trajectories.append(traj)
                idx += 1
                if idx == 5000:
                    break
            # calculate the self-bleu
            bleu_scores = []
            for i in range(len(gen_trajectories)):
                bleu_score = 0
                for j in range(len(gen_trajectories)):
                    if i == j:
                        continue
                    bleu_score += editdistance.eval(gen_trajectories[i], gen_trajectories[j])
                bleu_scores.append(bleu_score)

    return
    # Visualize clusters



for prefix in prefixs:
    model_res = {}
    plt.figure(figsize=(8, 6), dpi=300)
    pretrained_res = {"Pretrained": []}
    for model in models:
        if prefix == 'My_Transformer_od_ftgeolifeusr_':
            data_name = 'geo'
            other_models = other_models_full['geo']
        elif prefix == 'My_Transformer_od_fttdriveusr_':
            data_name = 'drive'
            other_models = other_models_full['drive']
        elif 'My_Transformer_od_ft#usr_' in prefix:
            data_name = 'toyota'
            other_models = other_models_full['toyota']
        with open(f'./final_res/od_trajectories_{data_name}.pk', 'rb') as f:
            od_trajectories = pickle.load(f)
        print(prefix + model)
        it = 'last'
        model_name = prefix + model + '_' + str(it)
        if prefix == 'My_Transformer_od_ft#usr_':
            for phase in range(11, 16):
                prefix_ = prefix.replace('#', str(phase))
                model_name = prefix_ + model  # + '_' + str(it)
                if model == 'pretrained':
                    model_name = 'pretrained'
                try:
                    with open(f'./final_res/{model_name}_full_gen_{data_name}.pk', 'rb') as f:
                        gen_trajectories = pickle.load(f)
                        print(f"{model}: {len(gen_trajectories)}")
                    # read the real trajectories
                    with open(f'./final_res/{model_name}_full_tar_{data_name}.pk', 'rb') as f:
                        real_trajectories = pickle.load(f)

                    break
                except:
                    continue
        else:
            try:
                model_name = prefix + model
                if model == 'pretrained':
                    model_name = 'pretrained'
                with open(f'./final_res/{model_name}_full_gen_{data_name}.pk', 'rb') as f:
                    gen_trajectories = pickle.load(f)
                    print(f"{model}: {len(gen_trajectories)}")
                # read the real trajectories
                with open(f'./final_res/{model_name}_full_tar_{data_name}.pk', 'rb') as f:
                    real_trajectories = pickle.load(f)
            except:
                # print(f"Model {model_name} not found")
                continue
        trajectories_set['truth'] = real_trajectories
        trajectories_set[model] = gen_trajectories

    for other in other_models:
        print(other)
        try:
            with open(f'./final_res/{other}.pk', 'rb') as f:
                gen_trajectories = pickle.load(f)
                print(f"{other}: {len(gen_trajectories)}")
            # read the real trajectories
            other_tar = other.replace('gen', 'tar')
            with open(f'./final_res/{other_tar}.pk', 'rb') as f:
                real_trajectories = pickle.load(f)
        except:
            # print(f"Model {model_name} not found")
            continue
        trajectories_set[other] = gen_trajectories

    selfbleu_trajectories(trajectories_set)
