import os
import random
import time

import torch

from Env.toyota import *
from IQL.models.iq_model_drive import *
from get_tdrive_dataset import get_rl_data
import types
from IQL.models.Memory import Memory
from eval_drive import eval_generation_agg


def experiment(variant):
    # Load configuration
    trajectories, test_trajectories, grid = get_rl_data(usr=variant['usr'])

    # build a dict to store the od and trajectory idx
    # od_to_traj = {}
    # for i, traj in enumerate(trajectories):
    #     od = "-".join([str(traj['observations'][0][0]), str(traj['observations'][0][1])])
    #     if od not in od_to_traj:
    #         od_to_traj[od] = []
    #     od_to_traj[od].append(i)
    # # remove the od with only one trajectory
    # for od in list(od_to_traj.keys()):
    #     if len(od_to_traj[od]) < 2:
    #         del od_to_traj[od]

    if variant.get('device', 'cuda').split(":")[0] == 'cuda' and torch.cuda.is_available():
        cuda_idx = int(variant.get('device', 'cuda').split(":")[1])
        torch.cuda.set_device(cuda_idx)
        device = torch.device(f'cuda:{cuda_idx}')
    # device = torch.device('cpu')

    from Env.tdrive import TdriveEnv
    if variant['usr'] == 'all':
        env = TdriveEnv(grid=grid, state_dim=7)
    else:
        env = TdriveEnv(grid=grid, state_dim=6)
    max_ep_len = variant.get('max_ep_len', 256)

    state_dim = env.observation_space.shape[0]
    act_dim = 10
    is_traj = True
    opt = {
        "obs_size": state_dim,
        'device': device,
        'lr': variant.get('learning_rate', 5e-5),
        'batch_size': variant.get('batch_size', 32),
        'hidden_size': variant.get('embed_dim', 256),
        'seq_len': max_ep_len,
        'max_iters': variant.get('max_iters', 1000),
        'critic_tau': 0.1,
        'is_traj': is_traj,
        'act_dim': act_dim,
    }
    print(opt)
    agent = SoftQ(opt)
    full_expert_buffer = Memory(500000, 0)
    full_expert_buffer.load(trajectories, 1, 1)
    data = full_expert_buffer.buffer
    step = 0
    state_dict_name = "iql_preference_global_drive200"
    try:
        state_dict = torch.load(f'./save_preference/{state_dict_name}.pth')
    except:
        state_dict = torch.load(f'./save_preference/{state_dict_name}.pth',map_location=torch.device('cuda'))
    model_state_dict = agent.q_net.state_dict()
    agent.q_net.load_state_dict(state_dict)
    now = time.time()
    eval_generation_agg(model=agent, trajectories=test_trajectories, link_to_id=grid,
                        state_dim=state_dim, act_dim=act_dim, device=device, state_mean=1,
                        state_std=1, env=env, scale=1, model_name="IQL")
    for e in range(opt['max_iters']):
        loss_set = {}
        # idx = np.random.permutation(len(data))
        idx = np.array(range(len(data)))
        agent.q_net.train()
        for j in range(int(len(data) / opt['batch_size'])):
        # for j in range(1000):
            expert_batch = [data[i] for i in idx[j * opt['batch_size']:(j + 1) * opt['batch_size']]]
            expert_batch = full_expert_buffer.get_samples(opt['batch_size'],
                                                          opt['device'], expert_batch, is_traj=opt['is_traj'])
            losses = agent.iq_update(expert_batch, step, None)
            for key, value in losses.items():
                if key not in loss_set:
                    loss_set[key] = []
                loss_set[key].append(value)
            step += 1

        print(f"Epoch {e} | Time: {time.time() - now}")
        for key, value in loss_set.items():
            print(f"Epoch {e} | {key}: {np.mean(value)}")

        # log_losses_to_file(loss_set, 'losses.txt')
        # continue
        if (e) % 20 == 0:
            # save model
            if e >= 100:
                torch.save(agent.q_net.state_dict(), f'./save_preference/iql_preference_global_drive{e}.pth')
                agent.q_net.eval()
            eval_generation_agg(model=agent, trajectories=test_trajectories, link_to_id=grid,
                                state_dim=state_dim, act_dim=act_dim, device=device, state_mean=1,
                                state_std=1, env=env, scale=1, model_name="IQL")
    torch.save(agent.q_net.state_dict(), f'./save_preference/iql_preference_global_drive{e}.pth')
    eval_generation_agg(model=agent, trajectories=test_trajectories, link_to_id=grid,
                        state_dim=state_dim, act_dim=act_dim, device=device, state_mean=1,
                        state_std=1, env=env, scale=1, model_name="IQL")
