# TRC-25-02834  

This project implements and evaluates multiple trajectory generation models with support for original and preference-based methods. **Only for peer review and complete project will be opensource online after publication**

## 📦 Requirements

- Python 3.8+
- PyTorch >= 1.10
- NumPy
- argparse

Install dependencies with:

```bash
pip install -r requirements.txt
````

## 🚀 Running the Project

The project supports several modes and model types:

```bash
python main.py --env toyota --dataset _ --mode normal --K 64 --pct_traj 1.0 \
--batch_size 64 --max_ep_len 256 --n_positions 256 \
--model_type my --embed_dim 512 --n_layer 3 --n_head 2 \
--activation_function relu --dropout 0.1 --learning_rate 1e-3 \
--weight_decay 1e-4 --warmup_steps 10000 --num_eval_episodes 100 \
--max_iters 10 --seed 100 --num_steps_per_iter 100 \
--device cuda:0 --usr all --ft_type rlhfw0
```

### 🔧 Key Arguments

| Argument       | Type  | Default | Description                                 |
| -------------- | ----- | ------- | ------------------------------------------- |
| `--env`        | str   | toyota  | Environment: `toyota`, `tdrive` , `porto`  |
| `--dataset`    | str   | \_      | Dataset identifier                          |
| `--mode`       | str   | normal  | Can be `normal` or `delayed`                |
| `--model_type` | str   | my      | `my`, `myp`, `mc`                           |
| `--ft_type`    | str   | rlhfw0  | Fine-tuning type                            |
| `--embed_dim`  | int   | 512     | Embedding dimension                         |
| `--n_layer`    | int   | 3       | Number of Transformer layers                |
| `--n_head`     | int   | 2       | Number of attention heads                   |
| `--dropout`    | float | 0.1     | Dropout rate                                |
| `--seed`       | int   | 100     | Random seed (multiplied by 1024 internally) |
| `--device`     | str   | cuda:0  | Target device (`cpu` or `cuda:0`)           |



