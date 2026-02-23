import argparse
import random
import numpy as np
import torch
import os
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='toyota') # tdrive, toyota
parser.add_argument('--dataset', type=str, default='_')  # data_2m
parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
parser.add_argument('--K', type=int, default=64)
parser.add_argument('--pct_traj', type=float, default=1.)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_ep_len', type=int, default=256)
parser.add_argument('--n_positions', type=int, default=256)
# my for original, myp for preference based
parser.add_argument('--model_type', type=str, default='mk')
parser.add_argument('--embed_dim', type=int, default=512)
parser.add_argument('--n_layer', type=int, default=3)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--activation_function', type=str, default='relu')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
parser.add_argument('--warmup_steps', type=int, default=10000)
parser.add_argument('--num_eval_episodes', type=int, default=100)
parser.add_argument('--max_iters', type=int, default=10)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--num_steps_per_iter', type=int, default=100)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--usr', type=str, default='all')
parser.add_argument('--ft_type', type=str, default='rlhfw0')



def set_seed(seed: int = 1) -> None:
    seed *= 1024
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # below option set seed for current gpu
    torch.cuda.manual_seed(seed)
    # below option set seed for ALL GPU
    # torch.cuda.manual_seed_all(seed)

    # When running on the CuDNN backend, two further options must be set
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed for training set as {seed}")


if __name__ == '__main__':
    args = parser.parse_args()
    set_seed(args.seed)
    if args.model_type in ["my", "myp", "mc"]:
        from Proposed.navigate import experiment
        print("My Experiment Started...")
        if args.model_type == "myp":
            opt = {
                "obs_size": 6,
                'device': args.device,
                'lr': 1e-3,
                'batch_size': 32,
                'hidden_size': 256,
                'seq_len': 256,
                'epochs': 2000,
                'critic_tau': 0.1,
                'is_train': False,
                'act_dim': args.act_dim,
                'is_traj': True,
            }
            from IQL.models.iq_model import SoftQ
            preference_model = SoftQ(opt)
            experiment('my', variant=vars(args), preference_model=preference_model)
        if args.model_type == "my":
            experiment('my', variant=vars(args))
    else:
        assert False
