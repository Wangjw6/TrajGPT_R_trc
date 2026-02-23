import numpy as np
import torch

import time


class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False, ref_model=None):

        train_losses = []
        auxlosses = []
        logs = dict()

        train_start = time.time()
        gradient_norms = []
        self.model.train()
        for step in range(num_steps):
            # print(f'Step {step}', end='\r')
            train_loss, gn, aux_loss = self.train_step(ref_model=ref_model)
            train_losses.append(train_loss)
            auxlosses.append(aux_loss)
            gradient_norms.append(gn)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/aux_loss_mean'] = np.mean(auxlosses)
        logs['training/train_loss_std'] = np.std(train_losses)
        logs['training/gradient_norm_mean'] = np.mean(gradient_norms)
        print(
            f'Iteration {iter_num} | time/total: {logs["time/total"]} | training/train_loss_mean: {logs["training/train_loss_mean"]} | training/train_loss_std: {logs["training/train_loss_std"]} | training/gradient_norm_mean: {logs["training/gradient_norm_mean"]}| training/aux_loss_mean: {logs["training/aux_loss_mean"]}')
        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        return logs



    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:, 1:], action_target, reward_target[:, 1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

