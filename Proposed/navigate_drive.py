
import copy
from eval_drive import eval_generation_agg
import random
from Proposed.models.models.my_transformer_od_awareness import My_Transformer_od_awareness
from Proposed.models.models.my_transformer_perference_awareness import My_Transformer_preference_awareness
from Proposed.models.training.seq_trainer import SequenceTrainer


from Env.toyota import *
import os
import time
import torch


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    bytes_size = total_params * 4  # Each parameter is 4 bytes (32 bits)
    mb_size = bytes_size / (1024 ** 2)  # Convert bytes to MB
    print(f"Model size: {mb_size:.4f} MB")


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    gamma = 1.
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
        preference_model=None,
):

    model_type = variant.get('model_type', 'my')
    device = variant.get('device', 'cpu')
    if variant.get('device', 'cuda').split(":")[0] == 'cuda' and torch.cuda.is_available():
        cuda_idx = int(variant.get('device', 'cuda').split(":")[1])
        torch.cuda.set_device(cuda_idx)
        device = torch.device(f'cuda:{cuda_idx}')
    if os.name == 'nt':
        device = torch.device('cuda')
    env_name, dataset = variant['env'], variant['dataset']
    assert env_name == 'tdrive'
    from Env.tdrive import TdriveEnv

    from get_tdrive_dataset import get_rl_data
    trajectories, test_trajectories, grid = get_rl_data(usr=variant['usr'])
    if variant['usr'] == 'all':
        env = TdriveEnv(state_dim=7, grid=grid)
    else:
        env = TdriveEnv(state_dim=6, grid=grid)

    max_ep_len = 256
    env_targets = [100, 50]
    scale = 10.
    state_dim = env.observation_space.shape[0]
    act_dim = 10
    action_counts = {}

    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        real_a = path['actions']
        for a in real_a:
            if a not in action_counts:
                action_counts[a] = 0
            action_counts[a] += 1
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())

    traj_lens, returns = np.array(traj_lens), np.array(returns)


    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1

    p_sample = np.ones(num_trajectories) / num_trajectories
    def get_batch(batch_size=32, max_len=K, step=-1):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )
        if step > 0:
            batch_inds = np.arange((step-2) * batch_size, (step-1)*batch_size)
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        action_mask_batch = []

        for i in range(batch_size):
            traj = trajectories[int(batch_inds[i])]

            si = random.randint(0, traj['rewards'].shape[0] - 1)
            max_len_ = min(max_len, traj['rewards'].shape[0] - si)
            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len_].reshape(1, -1, state_dim))
            a_mask = np.zeros((1, max_len_, act_dim))

            a.append(traj['actions'][si:si + max_len_].reshape(1, -1, 1))
            r.append(traj['rewards'][si:si + max_len_].reshape(1, -1, 1))
            action_mask_batch.append(a_mask)
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len_].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len_].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, 1)) * 0, a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            action_mask_batch[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), action_mask_batch[-1]],
                                                   axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        action_mask_batch = torch.from_numpy(np.concatenate(action_mask_batch, axis=0)).to(device=device)
        return s, a, r, d, rtg, timesteps, mask, action_mask_batch

    def get_batch_full(batch_size=32, max_len=K, step=-1):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,
        )
        if step > 0:
            batch_inds = np.arange((step-2) * batch_size, (step-1)*batch_size)
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        action_mask_batch = []
        state_action = []
        traj_lens = []
        for i in range(batch_size):
            traj = trajectories[int(batch_inds[i])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)
            max_len_ = min(max_len, traj['rewards'].shape[0] - si)
            # get full traj info
            sa = np.concatenate([traj['observations'].reshape(-1, 7), traj['actions'].reshape(-1, 1)], axis=1)
            state_action.append(sa)
            traj_lens.append([si, si + max_len_])
            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len_].reshape(1, -1, state_dim))
            a_mask = np.zeros((1, max_len_, act_dim))
            for k in range(s[-1].shape[1]):
                current_link = int(s[-1][0, k, 2])
                action_space = len(find_connect(id_to_link[current_link])) + 1
                a_mask[0, k, action_space:] = 1.
            a.append(traj['actions'][si:si + max_len_].reshape(1, -1, 1))
            r.append(traj['rewards'][si:si + max_len_].reshape(1, -1, 1))
            action_mask_batch.append(a_mask)
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len_].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len_].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=0.95)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            # print(rtg[-1])
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, 1)) * 0, a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            action_mask_batch[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), action_mask_batch[-1]],
                                                   axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        action_mask_batch = torch.from_numpy(np.concatenate(action_mask_batch, axis=0)).to(device=device)

        for i in range(len(state_action)):
            state_action[i] = np.concatenate(
                [state_action[i], np.zeros((1024 - len(state_action[i]), state_action[i].shape[1]))])
        state_action = np.array(state_action)
        state_action = torch.from_numpy(state_action).float().to(device=device)
        traj_lens = torch.from_numpy(np.array(traj_lens)).long().to(device=device)
        return s, a, r, d, rtg, timesteps, mask, action_mask_batch, state_action, traj_lens
    phase = variant.get('phase', '0')
    ref_model = None
    is_ft = 0
    flag = 0
    if model_type == "myp":
        assert preference_model is not None
        model_name = 'My_Transformer_od_ft' + env_name
        model = My_Transformer_preference_awareness(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4 * variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=variant['n_positions'],
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            normal_helper=action_counts,
            device=device,
            usr=variant['usr'],
            # preference_model=preference_model.q_net,
            env=env_name,
            ft_type=variant['ft_type'],
        )
        if '1' in phase:
            state_dict_name = 'My_Transformer_od_awarenesstdrive_0_00_380'#'My_Transformer_od_awarenesstdrive_0_219_0904'#My_Transformer_od_awarenesstdrive_0_239_0904'
            if variant['ft_type'] == 'dpo':
                state_dict_name = 'My_Transformer_od_fttdriveusr_dpo_116'
            elif variant['ft_type'] == 'rlhfw0':
                state_dict_name = 'My_Transformer_od_fttdriveusr_rlhfw0_236'
            else:
                flag = 1
            try:
                state_dict = torch.load(f'./saved_models/{state_dict_name}.pt')
            except:
                state_dict = torch.load(f'./saved_models/{state_dict_name}.pt', map_location={'cuda:1': 'cuda:0'})
            model_state_dict = model.state_dict()

            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

            model.load_state_dict(filtered_state_dict, strict=False)

            model.set_preference_model_fix(preference_model.q_net)
            preference_model_name = "iql_preference_global_drive200" #"Lsoftq1500"
            try:
                state_dict = torch.load(f'./save_preference/{preference_model_name}.pth')
            except:
                state_dict = torch.load(f'./save_preference/{preference_model_name}.pth', map_location=torch.device('cuda'))
            model_state_dict = model.preference_model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
            model.preference_model.load_state_dict(filtered_state_dict, strict=False)
        else:
            model.set_preference_model_fix(preference_model.q_net)
        ref_model = copy.deepcopy(model)
        model.ref_model = ref_model
    if model_type == "my":
        model_name = 'My_Transformer_od_awareness' + env_name
        model = My_Transformer_od_awareness(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4 * variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=variant['n_positions'],
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            normal_helper=action_counts,
            device=device,
            usr=variant['usr'],
            env=env_name,
            ft_type=variant['ft_type'],
        )
        if '1' in phase:
            state_dict_name = 'My_Transformer_od_awarenesstdrive_0_00_380'
            try:
                state_dict = torch.load(f'./saved_models/{state_dict_name}.pt')
            except:
                try:
                    state_dict = torch.load(f'./saved_models/{state_dict_name}.pt', map_location={'cuda:2': 'cuda:0'})
                except:
                    state_dict = torch.load(f'./saved_models/{state_dict_name}.pt', map_location={'cuda:1': 'cuda:0'})
            model_state_dict = model.state_dict()
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
            model.load_state_dict(filtered_state_dict, strict=True)

    model = model.to(device=device)
    try:
        model = torch.compile(model)
    except:
        print('Compile failed')
        pass
    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],

    )
    count_parameters(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )
    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch if preference_model is None else get_batch_full,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
        eval_fns=[None for tar in env_targets],
    )
    phase = variant.get('phase', '0')
    if variant['usr'] == 'all':
        model_name += 'usr'
    print(model_name)
    if '1' in phase:
        ft_type = variant['ft_type']
        saved_model_name = f"{model_name}_{ft_type}"
    else:
        saved_model_name = model_name
    if flag == 1:
        saved_model_name = "tdrive_pretrained"

    for iter in range(variant['max_iters']):
        print("Train")
        trainer.model.train()
        now = time.time()
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter + 1, print_logs=True, ref_model=ref_model)
        print(f"time cost: {time.time() - now}")

        if (is_ft == 1 and (iter) % 20 == 0) or (is_ft == 0 and iter % 20 == 0):
            if '1' in phase:
                ft_type=variant['ft_type']
                torch.save(trainer.model.state_dict(), f"./saved_models/{model_name}_{ft_type}_{iter}.pt")
            else:
                torch.save(trainer.model.state_dict(), f"./saved_models/{model_name}_{is_ft}_{phase}_{iter}.pt")
            trainer.model.eval()
            eval_generation_agg(model=trainer.model, trajectories=test_trajectories, link_to_id=link_to_id,
                                state_dim=state_dim, act_dim=act_dim, device=device, state_mean=state_mean,
                                state_std=state_std, env=env, scale=scale, model_name=saved_model_name+"_full")

            print()
    eval_generation_agg(model=trainer.model, trajectories=test_trajectories, link_to_id=link_to_id,
                        state_dim=state_dim, act_dim=act_dim, device=device, state_mean=state_mean,
                        state_std=state_std, env=env, scale=scale, model_name=saved_model_name+"_full")
    torch.save(trainer.model.state_dict(), f'./saved_models/{model_name}_{is_ft}_last.pt')
