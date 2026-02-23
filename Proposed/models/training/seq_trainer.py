import numpy as np
import torch
import torch.nn.functional as F
from Proposed.models.training.trainer import Trainer
import torch.nn as nn
import matplotlib.pyplot as plt


class SequenceTrainer(Trainer):
    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None):
        super().__init__(model, optimizer, batch_size, get_batch, loss_fn, scheduler, eval_fns)
        self.update_steps = 1

    def train_step(self, ref_model=None):
        self.update_steps += 1
        batch = self.get_batch(self.batch_size)
        if len(batch) == 8:
            states, actions, rewards, dones, rtg, timesteps, attention_mask, action_mask_batch = batch
            full_trajs, traj_lens = None, None
        else:
            states, actions, rewards, dones, rtg, timesteps, attention_mask, action_mask_batch, full_trajs, traj_lens = batch
        action_target = torch.clone(actions)
        state_preds, action_preds, aux_info_pred = self.model.forward(
            states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask,
            action_mask=action_mask_batch
        )
        r = 0
        loss2 = 0
        if full_trajs is not None:
            if ref_model is not None:
                _, old_action_preds, aux_info = ref_model.forward(
                    states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask,
                    action_mask=action_mask_batch
                )
                r = self.model.finetune_loss(states, full_trajs, traj_lens, action_preds, attention_mask,
                                               action_target, old_action_preds, aux=aux_info, update_steps=self.update_steps, dones=dones)

            loss2 = r
        act_dim = action_preds.shape[2]
        action_preds_ = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target_ = action_target.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        criterion = nn.CrossEntropyLoss()
        if self.model.counts_tensor is not None and (self.model.env == 'tdrive' or self.model.env == 'porto'):
            class_weights = self.model.counts_tensor.sum() / self.model.counts_tensor.reshape(-1)
            class_weights = class_weights / class_weights.sum()
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        loss = criterion(action_preds_, action_target_.to(torch.int64).reshape(-1))

        if ref_model is not None:
            final_loss = loss2
            if self.model.env == 'toyota':
                final_loss = loss2 * 0.02 + loss
            if self.model.env == 'tdrive' and 'rlhf' in self.model.ft_type:
                final_loss = loss2 * 0.02 + loss
        else:
            final_loss = loss
        self.optimizer.zero_grad()
        final_loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
        self.optimizer.step()
        torch.cuda.synchronize()
        if loss2 != 0:
            return loss.detach().cpu().item(), norm.detach().cpu().item(), loss2.detach().cpu().item()
        return loss.detach().cpu().item(), norm.detach().cpu().item(), 0

