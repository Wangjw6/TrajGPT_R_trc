"""
Copyright 2022 Div Garg. All rights reserved.

Standalone IQ-Learn algorithm. See LICENSE for licensing terms.
"""
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from torch.distributions import Categorical
import torch.nn as nn
from Proposed.models.models.preference_model import *


def soft_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


class encoder(nn.Module):
    def __init__(self, input_dim=3, z_dim=64, h_dim=256):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        # self.lstm_model = nn.LSTM(h_dim,
        #                           h_dim,
        #                           batch_first=True)
        self.input_layer = nn.Linear(input_dim, h_dim)
        self.fc = nn.Linear(h_dim, z_dim * 2)
        self.ln = nn.LayerNorm(z_dim * 2)

    def forward(self, x):
        # x = self.input_layer(x)
        x = self.input_layer(x)
        x = F.leaky_relu(x)
        # output, (hidden, cell) = self.lstm_model(x)
        # hidden = hidden.view(-1, self.h_dim)
        z = F.leaky_relu(self.fc(x))
        z = self.ln(z)
        mu, log_var = torch.chunk(z, 2, dim=1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class Critic(nn.Module):
    def __init__(self, device, obs_size=7, action_dim=9, seq_len=256, h_dim=256, is_traj=False, num_pref=6):
        super().__init__()
        # obs[0:2] is the position, obs[3] is the velocity, obs[4] is the step tag, obs[5] is the time tag
        self.h_dim = h_dim
        self.embed_link = nn.Embedding(121649, h_dim)
        self.embed_o = nn.Embedding(121649, h_dim)
        self.embed_d = nn.Embedding(121649, h_dim)
        self.embed_timestep = nn.Embedding(seq_len, h_dim)
        self.embed_departure = nn.Embedding(144, h_dim)
        # self.embed_speed = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(h_dim * 4, h_dim)
        self.out = nn.Linear(h_dim, action_dim)
        self.preference_token_embedding = nn.Embedding(10000, h_dim)

        self.preference_selector = nn.Linear(h_dim, num_pref)
        self.preference_bias = nn.Linear(h_dim, action_dim)
        self.device = device
        self.is_traj = is_traj

    def embed(self, x, traj=None):
        o, d, link, depart = x[:, 4], x[:, 5], x[:, 0], x[:, 3]
        usr_id = x[:, 6]
        # lat = x[:, 1]
        # lat = lat.to(torch.int64)
        # lon = lon.to(torch.int64)
        usr_id = usr_id.to(torch.int64)
        link = link.to(torch.int64)
        o = o.to(torch.int64)
        d = d.to(torch.int64)
        # speed = speed.to(torch.float32).reshape(-1, 1)
        depart = depart.to(torch.int64)
        embed_link = self.embed_link(link)
        embed_o = self.embed_o(o)
        embed_d = self.embed_d(d)
        embed_depart = self.embed_departure(depart)

        state_embeddings = torch.cat([embed_o, embed_d, embed_link, embed_depart], dim=-1)
        state_embeddings = self.embed_state(state_embeddings)
        state_embeddings = F.leaky_relu(state_embeddings)
        usr_embeddings = self.preference_token_embedding(usr_id)
        preference_embedding = embed_o + embed_d + embed_depart + usr_embeddings
        return state_embeddings, preference_embedding

    def forward(self, x, traj=None):
        state_embeddings, preference_embedding = self.embed(x, traj)
        assert preference_embedding is not None, "Preference must work!!!"
        return self.out(state_embeddings), preference_embedding, self.preference_bias(preference_embedding)


class SoftQ(object):
    def __init__(self, opt):
        self.gamma = 0.9
        self.batch_size = opt['batch_size']
        self.device = torch.device(opt['device'])
        self.actor = None
        self.is_traj = opt['is_traj']
        self.critic_target_update_frequency = 4
        self.log_alpha = torch.tensor(0.01, requires_grad=False,
                                      device=self.device)
        self.q_net = Critic(action_dim=opt['act_dim'], device=self.device, is_traj=opt['is_traj'],
                            h_dim=opt['hidden_size']).to(self.device)
        self.target_net = Critic(action_dim=opt['act_dim'], device=self.device, is_traj=opt['is_traj'],
                                 h_dim=opt['hidden_size']).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.critic_tau = opt['critic_tau']
        self.critic_optimizer = Adam(self.q_net.parameters(), lr=opt["lr"])
        self.train()


    def train(self, training=True):
        self.training = training
        self.q_net.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def critic_net(self):
        return self.q_net

    @property
    def critic_target_net(self):
        return self.target_net

    def choose_action(self, state, action_mask):
        with torch.no_grad():
            q, _, _ = self.q_net(state)
            dist = F.softmax(q / self.alpha, dim=1)

            dist = Categorical(dist)
            action = dist.sample()
            entropy = dist.entropy()
        return action

    def getV(self, obs, traj=None):
        if traj is not None:
            q, preference, q_ = self.q_net(obs, traj)
            q += q_
            return self.alpha * \
                torch.logsumexp(q / self.alpha, dim=1, keepdim=True)
        q, _, _ = self.q_net(obs, traj)
        v = self.alpha * \
            torch.logsumexp(q / self.alpha, dim=1, keepdim=True)
        return v

    def critic(self, obs, action=None, traj=None):

        if traj is not None:
            q, preference, q_ = self.q_net(obs, traj) # yes
            if action is not None:
                return q.gather(1, action.long()), preference, q_.gather(1, action.long())
            return q, preference, q_
        q, _, _ = self.q_net(obs)
        return q.gather(1, action.long()), None, None

    def get_full_q(self, obs, traj=None):
        if traj is not None:
            q, preference, q1 = self.q_net(obs, traj)
            return q, preference, q1
        q, _, _ = self.q_net(obs)
        return q, None, None

    def get_targetV(self, obs, traj=None):
        if traj is not None:
            q, preference, q_ = self.target_net(obs, traj) # yes
            q += q_
            return self.alpha * \
                torch.logsumexp(q / self.alpha, dim=1, keepdim=True)
        q, _, _ = self.target_net(obs)
        target_v = self.alpha * \
                   torch.logsumexp(q / self.alpha, dim=1, keepdim=True)
        return target_v

    def iq_update(self, expert_batch, step, od_expert_batch=None):
        losses = self.iq_update_critic(expert_batch, od_expert_batch)

        if step % self.critic_target_update_frequency == 0:
            soft_update(self.critic_net, self.critic_target_net,
                        self.critic_tau)
        return losses

    # Full IQ-Learn objective with other divergences and options
    def iq_loss(self, current_Q, current_v, next_v, batch, traj=None):
        gamma = self.gamma
        obs, next_obs, action, env_reward, done = batch

        loss_dict = {}
        # keep track of value of initial states
        is_expert = torch.ones(obs.shape[0], dtype=torch.bool, device=self.device)
        v0 = self.getV(obs[is_expert, ...], traj).mean()
        loss_dict['v0'] = v0.item()

        #  calculate 1st term for IQ loss
        #  -E_(ρ_expert)[Q(s, a) - γV(s')]
        y = (1 - done) * gamma * next_v
        reward = (current_Q - y)[is_expert]


        phi_grad = 1
        # phi_grad = torch.exp(-reward) / (2 - torch.exp(-reward))
        loss = -(phi_grad * reward).mean() ##

        # check if nan in phi_grad, if so print the index
        # if torch.isnan(phi_grad).any():
        #     print('nan in phi_grad')
        #     print(torch.where(torch.isnan(phi_grad)))

        loss_dict['softq_loss'] = loss.item()

        # calculate 2nd term for IQ loss, we show different sampling strategies
        value_loss = (current_v - y)[is_expert].mean()
        loss += value_loss
        loss_dict['value_loss'] = value_loss.item()

        chi2_loss = 0.5 * (reward ** 2).mean()

        loss_dict['reg_loss'] = chi2_loss.item()

        loss += chi2_loss
        loss_dict['total_loss'] = loss.item()
        return loss, loss_dict

    # def iq_learn_update(self, policy_batch, expert_batch, logger, step):
    #     args = self.args
    #     # policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    #     obs, next_obs, action, reward, done = expert_batch
    #
    #     # if args.only_expert_states:
    #     #     expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done
    #
    #     # obs, next_obs, action, reward, done, is_expert = get_concat_samples(
    #     #     policy_batch, expert_batch, args)
    #
    #     loss_dict = {}
    #
    #     ######
    #     # IQ-Learn minimal implementation with X^2 divergence (~15 lines)
    #     # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
    #     current_Q = self.critic(obs, action)
    #     y = (1 - done) * self.gamma * self.getV(next_obs)
    #     if args.train.use_target:
    #         with torch.no_grad():
    #             y = (1 - done) * self.gamma * self.get_targetV(next_obs)
    #
    #     reward = (current_Q - y)
    #     loss = -(reward).mean()
    #
    #     # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
    #     value_loss = (self.getV(obs) - y).mean()
    #     loss += value_loss
    #
    #     # Use χ2 divergence (adds a extra term to the loss)
    #     chi2_loss = 1 / (4 * args.method.alpha) * (reward ** 2).mean()
    #     loss += chi2_loss
    #     ######
    #
    #     self.critic_optimizer.zero_grad()
    #     loss.backward()
    #     self.critic_optimizer.step()
    #     return loss

    def iq_update_critic(self, expert_batch, od_expert_batch=None):
        # args = self.args
        # policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
        expert_full, expert_len = None, None
        try:
            expert_obs, expert_next_obs, expert_action, expert_reward, expert_done, expert_full, expert_len = expert_batch
        except:
            expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch

        obs, next_obs, action = expert_batch[0:3]
        if expert_full is not None:
            traj = [expert_full, expert_len]
        else:
            traj = None
        current_V = self.getV(obs, traj)
        with torch.no_grad():
            next_V = self.get_targetV(next_obs, traj)
        obs_arr = obs.detach().cpu().numpy()
        q, preference, q_ = self.critic(obs, None, traj)
        q_1 = q.gather(1, action.long())
        q_2 = q_.gather(1, action.long())
        if q_2 is not None:
            current_Q = q_1 + q_2
        else:
            current_Q = q_1
        critic_loss, loss_dict = self.iq_loss(current_Q, current_V, next_V, expert_batch[:5], traj)
        # if preference is not None:
        #     q_next, preference_next, q_next_ = self.critic(next_obs, None, traj)
        #     # print(torch.max(q_-q_next_))
        #     # print(torch.max(q-q_next))
        #     smoothness_loss = F.mse_loss(q_, q_next_.detach())
        #     critic_loss += smoothness_loss * 0.1
        self.critic_optimizer.zero_grad()

        critic_loss.backward()
        # print grad
        # for name, param in self.q_net.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient of {name} at one step:\n{param.grad.max()}")
        norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.)
        # step critic
        self.critic_optimizer.step()
        return loss_dict

    def get_counterfact_Q(self, obs, traj):
        q_1, ml, q_2 = self.get_full_q(obs, traj)
        return q_1+q_2
    #
    # def get_counterfact_loss(self, obs, action, reward, next_obs, done, traj):
    #     current_Q, ml, preference = self.critic(obs, action, traj)
    #     current_v = self.getV(obs, traj)
    #     next_v = self.get_targetV(next_obs, traj)
    #     loss, loss_dict = self.iq_loss(current_Q, current_v, next_v, [obs, next_obs, action, reward, done])
    #     return loss, loss_dict


if __name__ == '__main__':
    agent = SoftQ()
