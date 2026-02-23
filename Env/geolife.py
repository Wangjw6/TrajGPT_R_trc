import pickle
import gym
import numpy as np

# with open(f"./offline_data_hub/opendata/tdrive_vehiclesS.pkl", 'rb') as file:
#     trajs = pickle.load(file)
with open(f"./offline_data_hub/opendata/geolife_action_dict.pkl", 'rb') as file:
    geolife_action_dict = pickle.load(file)

with open(f"./offline_data_hub/opendata/geolife_inverse_action_dict.pkl", 'rb') as file:
    geolife_inverse_action_dict = pickle.load(file)


class GeolifeEnv(gym.Env):
    def __init__(self, train_data=None, test_data=None, grid=None, state_dim=6):
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
        self.max_actions = 10
        self.train_data = train_data
        self.test_data = test_data
        self.tdrive_inverse_action_dict = geolife_inverse_action_dict
        self.tdrive_action_dict = geolife_action_dict
        self.grid = grid
        self.state_dim = state_dim

    def step(self, obs, action):
        action = int(action)
        if action == 9:
            flag = 1
            return np.array(obs), 0., flag, []
        lon_shift = int(self.tdrive_inverse_action_dict[action].split('_')[1])
        lat_shift = int(self.tdrive_inverse_action_dict[action].split('_')[0])
        current_grid = int(obs[0])
        current_lat = int(obs[1])
        current_lon = int(obs[2])
        next_lat = current_lat + int(lat_shift)
        next_lon = current_lon + int(lon_shift)

        # print(next_lat, next_lon, self.tdrive_inverse_action_dict[action].split('_'))
        try:
            if (next_lat, next_lon) not in self.grid or next_lat <0 or next_lat > 2000 or next_lon < 0 or next_lon > 2000:
                next_obs = np.array([current_grid, current_lat, current_lon, obs[3], obs[4], obs[5]])
                if self.state_dim == 7:
                    next_obs = np.array([current_grid, next_lat, next_lon, obs[3], obs[4], obs[5], obs[6]])
                r = 0.
                flag = 1
                return next_obs, r, 0, None
        except:
            print(next_lat, next_lon)
            raise
        next_grid = self.grid[(next_lat, next_lon)]
        if self.state_dim == 7:
            next_obs = np.array([next_grid, next_lat, next_lon, obs[3], obs[4], obs[5], obs[6]])
        else:
            next_obs = np.array([next_grid, next_lat, next_lon, obs[3], obs[4], obs[5]])

        r = 0.

        return next_obs, r, 0, None

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