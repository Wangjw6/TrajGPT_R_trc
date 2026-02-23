import pickle
import gym
import numpy as np

# with open(f"./offline_data_hub/data_2021101_1107_tokyo_core/conn_dictS_heart2.pkl", 'rb') as file:
#     # Pickle the 'data' dictionary using the highest protocol available.
#     conn_dict = pickle.load(file)
# with open(f"./offline_data_hub/data_2021101_1107_tokyo_core/conn_dictS.pkl", 'rb') as file:
#     # Pickle the 'data' dictionary using the highest protocol available.
#     full_condict = pickle.load(file)

with open(f"./offline_data_hub/data_2m/conn_dictS.pkl", 'rb') as file:
    links = pickle.load(file)

conn_dict = links
full_condict = links

link_to_id = {}
for k, v in full_condict.items():
    if k not in link_to_id:
        link_to_id[k] = len(link_to_id)
    for c in v:
        if c not in link_to_id:
            link_to_id[c] = len(link_to_id)
id_to_link = {v: k for k, v in link_to_id.items()}
def find_connect(link, lat=None, lng=None):
    # for k, v in conn_dict.items():
    #     assert k != 87877393
    #     for vv in v:
    #         assert vv != 87877393
    link = int(link)
    if lat is None or lng is None:
        if link not in conn_dict:
            return []
        else:
            return conn_dict[link]
    else:
        return []


class ToyotaEnv(gym.Env):
    def __init__(self, train_data=None, test_data=None, state_dim=6):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(state_dim,), dtype=np.float32)
        self.state = np.zeros(2)
        self.goal = np.zeros(2)
        self.goal[0] = 1.0
        self.goal[1] = 1.0
        self.step_count = 0
        self.max_steps = 100
        # For TrajGAIL
        self.states = []
        self.max_actions = 20
        self.train_data = train_data
        self.test_data = test_data
        self.id_to_link = id_to_link
        self.state_dim = state_dim


    def seed(self, seed=0):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, obs, action):
        action = np.array(action).astype(int)
        current_link = int(obs[2])
        try:
            connect_links = find_connect(id_to_link[current_link])
        except Exception as e:
            return obs, 0., True, connect_links
        # print(connect_links)
        # if len(connect_links) == 0:
        #     print('no connect links')
        r = 0.
        flag = False
        if len(connect_links) > action:
            # print([link_to_id[l] for l in connect_links], action)
            next_link = connect_links[action]
            r = 0.
        else:
            action = max(0, action % len(connect_links) - 1)
            try:
                next_link = connect_links[action]
            except:
                # print('action', action)
                # print(len(connect_links))

                return obs, 0., False, connect_links

        try:
            next_link = link_to_id[next_link]
            if next_link>= 262144:
                flag = True
        except:
            flag = True
        try:
            next_obs = [int(obs[0]), int(obs[1]), next_link, obs[3], int(obs[4])+1, int(obs[5])]
        except:
            next_obs = [int(obs[0]), int(obs[1]), next_link, obs[3], int(obs[4]) + 1]
        if self.state_dim == 7:
            next_obs += [int(obs[6])]
        # next_obs = [obs[0][0], obs[0][1], next_link, obs[0][3], obs[0][4] + 1, obs[0][5]]
        connect_links = [link_to_id[l] for l in connect_links]
        return next_obs, r, flag, connect_links

    def reset(self):
        self.state = np.zeros(2)
        self.step_count = 0
        return self.state

    def render(self, mode='human'):
        print(self.state)

    def init(self, flag=-1):
        if flag == -1:
            idx = np.random.randint(0, len(self.train_data))
            return self.train_data[idx][0][:-1], self.train_data[idx][0][2]
        else:
            return self.test_data[flag][0][:-1], self.test_data[flag][0][2]