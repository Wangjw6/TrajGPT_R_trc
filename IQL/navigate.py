import os
import random
import time

import torch

from Env.toyota import *
from IQL.models.iq_model import *
from prepare_toyota_dataset import get_rl_data
import types
from IQL.models.Memory import Memory
from eval import eval_generation_agg



def experiment(variant):
    # Load configuration
    trajectories, test_trajectories, link_to_id = get_rl_data(usr=variant['usr'])
    # dump trajectories and the time
    trajectories_ = []
    for i in range(len(test_trajectories)):
        real_trajectory = []
        for j in range(test_trajectories[i]['observations'].shape[0]):
            real_trajectory.append(test_trajectories[i]['observations'][j][2])
        trajectories_.append([real_trajectory, [test_trajectories[i]['observations'][0][k] for k in range(test_trajectories[i]['observations'][0].shape[0])]])
    with open(f'./final_res/trajectories_toyota_with_tag.pk', 'wb') as f:
        pickle.dump(trajectories_, f)
    print("user setting: ", variant['usr'])
    # build a dict to store the od and trajectory idx
    od_to_traj = {}
    for i, traj in enumerate(trajectories):
        od = "-".join([str(traj['observations'][0][0]), str(traj['observations'][0][1])])
        if od not in od_to_traj:
            od_to_traj[od] = []
        od_to_traj[od].append(i)
    # remove the od with only one trajectory
    for od in list(od_to_traj.keys()):
        if len(od_to_traj[od]) < 2:
            del od_to_traj[od]

    if variant.get('device', 'cuda').split(":")[0] == 'cuda' and torch.cuda.is_available():
        cuda_idx = int(variant.get('device', 'cuda').split(":")[1])
        torch.cuda.set_device(cuda_idx)
        device = torch.device(f'cuda:{cuda_idx}')
    # device = torch.device('cpu')

    from Env.toyota import ToyotaEnv
    env = ToyotaEnv(state_dim=7)
    max_ep_len = variant.get('max_ep_len', 256)

    state_dim = env.observation_space.shape[0]
    act_dim = variant.get('act_dim', 9)
    is_traj = True
    opt = {
        "obs_size": state_dim,
        'device': device,
        'lr': 5e-5,
        'batch_size': variant.get('batch_size', 32),
        'hidden_size': 256,
        'seq_len': max_ep_len,
        'epochs': variant.get('epochs', 1000),
        'critic_tau': 0.1,
        'is_traj': is_traj,
        'act_dim': act_dim,
    }
    print(opt)
    agent = SoftQ(opt)
    od_exp_buffer = {}
    for od, traj_idx in od_to_traj.items():
        expert_buffer = Memory(128, 0)
        expert_buffer.load([trajectories[i] for i in traj_idx], 1, 1)
        od_exp_buffer[od] = expert_buffer
    full_expert_buffer = Memory(1000000, 0)
    full_expert_buffer.load(trajectories, 1, 1)
    step = 0
    # try:
    #     agent.q_net.load_state_dict(torch.load(f'./save_preference/Lsoftq999.pth'))
    #     print("load model")
    # except:
    #     agent.q_net.load_state_dict(
    #         torch.load(f'./save_preference/Lsoftq999.pth', map_location=torch.device('cuda')))
    #     print("load model")
    state_dict_name = "iql_preference_global700"
    try:
        state_dict = torch.load(f'./save_preference/{state_dict_name}.pth')
    except:
        state_dict = torch.load(f'./save_preference/{state_dict_name}.pth',map_location=torch.device('cuda'))
    model_state_dict = agent.q_net.state_dict()
    # compare the keys of the two state_dict
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

    agent.q_net.load_state_dict(filtered_state_dict, strict=False)
    eval_generation_agg(model=agent, trajectories=test_trajectories, link_to_id=link_to_id,
                        state_dim=state_dim, act_dim=act_dim, device=device, state_mean=1,
                        state_std=1, env=env, scale=1, model_name="IQL")
    #

    data = full_expert_buffer.buffer

    now = time.time()
    if os.path.exists('losses.txt'):
        os.remove('losses.txt')
    for e in range(opt['epochs']):
        loss_set = {}
        # idx = np.random.permutation(len(data))
        idx = np.array(range(len(data)))
        agent.q_net.train()
        for j in range(int(len(data) / opt['batch_size'])):
            expert_batch = [data[i] for i in idx[j * opt['batch_size']:(j + 1) * opt['batch_size']]]
            expert_batch = full_expert_buffer.get_samples(opt['batch_size'],
                                                     opt['device'], expert_batch, is_traj=opt['is_traj'])
            # # sample an od
            # od = random.choice(list(od_exp_buffer.keys()))
            # od_expert_batch = od_exp_buffer[od].buffer
            # od_expert_batch = od_exp_buffer[od].get_samples(opt['batch_size'],
            #                                          opt['device'], od_expert_batch, is_traj=opt['is_traj'])
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
        if (e) % 50 == 0:
            # save model
            if e >= 0:
                torch.save(agent.q_net.state_dict(), f'./save_preference/iql_preference_global{e}.pth')
                agent.q_net.eval()
                eval_generation_agg(model=agent, trajectories=test_trajectories, link_to_id=link_to_id,
                                state_dim=state_dim, act_dim=act_dim, device=device, state_mean=1,
                                state_std=1, env=env, scale=1, model_name="IQL")
    torch.save(agent.q_net.state_dict(), f'./save_preference/iql_preference_global{e}_last.pth')
    eval_generation_agg(model=agent, trajectories=test_trajectories, link_to_id=link_to_id,
                        state_dim=state_dim, act_dim=act_dim, device=device, state_mean=1,
                        state_std=1, env=env, scale=1, model_name="IQL")

