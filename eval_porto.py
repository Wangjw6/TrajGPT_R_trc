import pickle
import time
import os
from datetime import datetime
import nltk
import numpy as np
import torch
import scipy.stats
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from Env.porto import *
import warnings
# Suppress specific warnings related to BLEU score calculation
warnings.filterwarnings("ignore", category=UserWarning, module='nltk.translate.bleu_score')

chencherry = SmoothingFunction()

def calculate_self_bleu(generated_sentences):
    chencherry = SmoothingFunction()  # Use smoothing function for better handling
    bleu_scores = []

    for i, candidate in enumerate(generated_sentences):
        references = [generated_sentences[j] for j in range(len(generated_sentences)) if j != i]
        score = sentence_bleu(references, candidate, smoothing_function=chencherry.method1)
        bleu_scores.append(score)

    return torch.tensor(bleu_scores).mean().item()  # Mean BLEU score
def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

import numpy as np
from collections import Counter
from math import sqrt
def cosine_similarity(list1, list2):
    vec1, vec2 = Counter(list1), Counter(list2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = sqrt(sum1) * sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def hamming_distance(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must be of the same length")
    return sum(el1 != el2 for el1, el2 in zip(list1, list2))


def levenshtein_distance(list1, list2):
    size_x, size_y = len(list1) + 1, len(list2) + 1
    matrix = np.zeros((size_x, size_y))

    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if list1[x - 1] == list2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return matrix[size_x - 1, size_y - 1]


def overlap_coefficient(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = len(set1.intersection(set2))
    return intersection / min(len(set1), len(set2))





def get_similarity(gen, real, func, func_name):
    sim = []
    for l1, l2 in zip(gen, real):
        if func_name == "BLEU":
            sim.append(func(l1, real))
        else:
            sim.append(func(l1, l2))
    mean_sim = np.array(sim).mean()
    print(f"{func_name}: {mean_sim}")
    return sim

def arr_to_distribution(arr, Min, Max, bins, over=None):
    """
    convert an array to a probability distribution
    :param arr: np.array, input array
    :param min: float, minimum of converted value
    :param max: float, maximum of converted value
    :param bins: int, number of bins between min and max
    :return: np.array, output distribution array
    """

    distribution, base = np.histogram(arr[arr <= Max], bins=bins, range=(Min, Max))
    m = np.array([len(arr[arr > Max])], dtype='int64')
    distribution = np.hstack((distribution, m))

    return distribution, base[:-1]


def get_js_divergence(p1, p2):
    """
    calculate the Jensen-Shanon Divergence of two probability distributions
    :param p1:
    :param p2:
    :return:
    """
    # normalize
    p1 = p1 / (p1.sum() + 1e-9)
    p2 = p2 / (p2.sum() + 1e-9)
    m = (p1 + p2) / 2
    js = 0.5 * scipy.stats.entropy(p1, m) + 0.5 * scipy.stats.entropy(p2, m)
    return js


def acc_calculate(real, pred):
    acc = 0
    total = 0
    for i in range(min(len(real), len(pred))):
        total += len(real[i])
        for j in range(len(real[i])):
            if j < len(pred[i]) and real[i][j] == pred[i][j]:
                acc += 1
    return acc / total
def conn_jsd(p1, p2, sub=2):
    link_dict = {}
    for u in p1:
        for i in range(len(u)-sub):
            p = u[i:i+sub]
            p = '-'.join([str(pp) for pp in p])
            if p not in link_dict:
                link_dict[p] = len(link_dict)
    for u in p2:
        for i in range(len(u)-sub):
            p = u[i:i+sub]
            p = '-'.join([str(pp) for pp in p])
            if p not in link_dict:
                link_dict[p] = len(link_dict)
    f, r = [], []
    for u in p1:
        for i in range(len(u)-sub):
            p = u[i:i+sub]
            p = '-'.join([str(pp) for pp in p])
            f.append(link_dict[p])
    for u in p2:
        for i in range(len(u)-sub):
            p = u[i:i+sub]
            p = '-'.join([str(pp) for pp in p])
            r.append(link_dict[p])

    MIN = np.min(r + f)
    MAX = np.max(r + f)
    # set st_loc_jsd
    bins = 400
    r = (np.array(r) - MIN) / (MAX - MIN)
    f = (np.array(f) - MIN) / (MAX - MIN)
    r_list, _ = arr_to_distribution(np.array(r), 0, 1, bins)
    f_list, _ = arr_to_distribution(np.array(f), 0, 1, bins)

    JSD = get_js_divergence(r_list, f_list)

    return JSD

def link_jsd(p1, p2):
    link_dict = {}
    for u in p1:
        for i in u:
            if str(i) not in link_dict:
                link_dict[str(i)] = len(link_dict)
    for u in p2:
        for i in u:
            if str(i) not in link_dict:
                link_dict[str(i)] = len(link_dict)
    f, r = [], []
    for u in p1:
        for i in u:
            f.append(link_dict[str(i)])
    for u in p2:
        for i in u:
            r.append(link_dict[str(i)])

    MIN = np.min(r + f)
    MAX = np.max(r + f)
    # set st_loc_jsd
    bins = 400
    r = (np.array(r) - MIN) / (MAX - MIN)
    f = (np.array(f) - MIN) / (MAX - MIN)
    r_list, _ = arr_to_distribution(np.array(r), 0, 1, bins)
    f_list, _ = arr_to_distribution(np.array(f), 0, 1, bins)

    JSD = get_js_divergence(r_list, f_list)

    return JSD


def bleu_score(candidate, od_trajectories):
    origin = int(candidate[0])
    destination = int(candidate[-1])
    od = f"{origin}_{destination}"
    #     print(od)
    if od not in od_trajectories:
        return 0
    reference = od_trajectories[od]
    return sentence_bleu(reference, candidate)


def eval_generation_agg(model, trajectories, link_to_id, state_dim, act_dim, device, state_mean, state_std, env, scale,
                        model_name=""):
    sum_reward = 0.
    finished = 0.
    action_counts = {}
    # extract real trajectories
    real_trajectories = []
    od_trajectories = {}
    record = []
    idx = []
    for i in range(len(trajectories)):
        real_trajectory = []
        for j in range(trajectories[i]['observations'].shape[0]):
            real_trajectory.append(int(trajectories[i]['observations'][j][0]))

        real_trajectories.append(real_trajectory)
        real_a = trajectories[i]['actions']

        # count action frequency
        for a in real_a:
            if a not in action_counts:
                action_counts[a] = 1
            else:
                action_counts[a] += 1
        origin = int(real_trajectory[0])
        destination = int(real_trajectory[-1])
        if f"{origin}_{destination}" not in od_trajectories:
            od_trajectories[f"{origin}_{destination}"] = []
        od_trajectories[f"{origin}_{destination}"].append(real_trajectory)
        if len(real_trajectories) <= 1000:
            idx.append(i)


    gen_trajectories = []
    action_record = {}
    action_entropy = []
    print(f"Generating trajectory...")
    for i in idx:
        action_list = []
        gen_trajectory = []
        observation = trajectories[i]['observations'][0].reshape(1, -1)
        states = torch.from_numpy(observation).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        cur_state = states
        actions = torch.zeros((0, 1), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        ep_return = 1.
        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1) / scale
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        episode_return, episode_length = 0, 0
        t = 0

        loss = []
        pred_action = []
        for j in range(len(real_trajectories[i])):
            action_mask = torch.ones(act_dim, device=device)
            gen_trajectory.append(int(observation[0][0]))
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            if model is None:
                action = np.random.randint(act_dim)
            elif model_name == "IQL":
                action = model.choose_action(
                    cur_state, action_mask
                )
            else:
                action = model.get_action(states,
                                          actions,
                                          rewards,
                                          target_return,
                                          timesteps,
                                          )

                entropy = -torch.sum(action * torch.log(action + 1e-9)).cpu().detach().numpy()
                action_entropy.append(entropy)
                pred_action.append(action)
                action = action.argmax()

            if j == 0:
                actions = action.reshape(1, -1)
            else:
                actions = torch.cat([actions, action.reshape(1, -1)], dim=0)

            if int(action) not in action_record:
                action_record[int(action)] = 1
            else:
                action_record[int(action)] += 1
            if action == 9:
                break
            action_list.append(int(action))
            next_observation, reward, flag, conn = env.step(observation.reshape(-1, ), action)
            observation = np.array(next_observation).reshape(1, -1)
            cur_state = torch.from_numpy(observation).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            pred_return = target_return[0, -1] - (reward / scale)

            target_return = torch.cat(
                [target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                 torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)
            t += 1
            episode_return += reward
            episode_length += 1

        if gen_trajectory == real_trajectories[i]:
            sum_reward += 1.
        if gen_trajectory[-1] == real_trajectories[i][-1]:
            finished += 1

        gen_trajectories.append(gen_trajectory)

    with open(f'./outputs/{model_name}_gen_porto.pk', 'wb') as f:
        pickle.dump(gen_trajectories, f)
    with open(f'./outputs/{model_name}_tar_porto.pk', 'wb') as f:
        pickle.dump(real_trajectories, f)
    print(f'time: {time.localtime()}')
    print(f'sum_reward: {sum_reward}')
    print(f'finished: {finished / len(real_trajectories)}')
    similarity = get_similarity(gen_trajectories, real_trajectories, jaccard_similarity, "Jaccard")
    similarity = get_similarity(gen_trajectories, real_trajectories, cosine_similarity, "Cosine")
    similarity = get_similarity(gen_trajectories, od_trajectories, bleu_score, "BLEU")

