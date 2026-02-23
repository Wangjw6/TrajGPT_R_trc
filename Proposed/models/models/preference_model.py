import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Proposed.models.models.tools import *
from prepare_toyota_dataset import get_rl_data
from itertools import chain


class obs_action_encoder(nn.Module):
    def __init__(self, obs_size=7, action_dim=9, seq_len=256, h_dim=256):
        super().__init__()
        # obs[0:2] is the position, obs[3] is the velocity, obs[4] is the step tag, obs[5] is the time tag
        self.h_dim = h_dim
        self.embed_link = nn.Embedding(262144, h_dim)
        self.embed_timestep = nn.Embedding(seq_len, h_dim)
        self.embed_departure = nn.Embedding(144, h_dim)
        self.embed_speed = torch.nn.Linear(1, h_dim)
        self.embed_action = torch.nn.Linear(action_dim, h_dim)

    def forward(self, x):
        o, d, link, speed, t, depart = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3], x[:, :, 4], x[:, :, 5]
        o = o.to(torch.int64)
        d = d.to(torch.int64)
        link = link.to(torch.int64)
        t = t.to(torch.int64)
        speed = speed.to(torch.float32).reshape(-1, speed.shape[1], 1)
        depart = depart.to(torch.int64)
        embed_o = self.embed_link(o)
        embed_d = self.embed_link(d)
        embed_link = self.embed_link(link)
        embed_speed = self.embed_speed(speed)
        embed_depart = self.embed_departure(depart)
        embed_timestep = self.embed_timestep(t)
        state_embeddings = embed_o + embed_d + embed_link + embed_speed + embed_depart + embed_timestep
        state_embeddings = F.leaky_relu(state_embeddings)
        return state_embeddings


class encoder(nn.Module):
    def __init__(self, z_dim=128, h_dim=256):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.lstm_model = nn.LSTM(h_dim,
                                  h_dim,
                                  batch_first=True)
        self.input_layer = nn.Linear(6, h_dim)
        self.fc = nn.Linear(h_dim, z_dim * 2)
        self.ln = nn.LayerNorm(z_dim * 2)

    def forward(self, x):
        x = self.input_layer(x)
        output, (hidden, cell) = self.lstm_model(x)
        hidden = hidden.view(-1, self.h_dim)
        z = F.leaky_relu(self.fc(hidden))
        z = self.ln(z)
        mu, log_var = torch.chunk(z, 2, dim=1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std



# class dc_decoder(nn.Module):
#     def __init__(self, z_dim, h_dim=256, seq_len=256, obs_size=7, action_size=1, action_dim=13):
#         super().__init__()
#         self.h_dim = h_dim
#         self.seq_len = seq_len
#         self.obs_size = obs_size
#         self.action_size = action_size
#         self.lstm = nn.LSTM(z_dim, h_dim, batch_first=True)
#         # self.output_layer = nn.Linear(h_dim, (self.obs_size + self.action_size) * self.seq_len)
#         self.o_recon = nn.Linear(h_dim, 262144)
#         self.d_recon = nn.Linear(h_dim, 262144)
#         self.speed_recon = nn.Linear(h_dim, 1 * seq_len)
#         self.t_recon = nn.Linear(h_dim, seq_len * seq_len)
#         self.depart_recon = nn.Linear(h_dim, 144)
#         self.a_recon = nn.Linear(h_dim, action_dim * seq_len)
#         # self.link_recon = nn.Linear(h_dim, 28000 * seq_len)
#
#     def forward(self, x):
#         output, _ = self.lstm(x)
#         output = output.reshape((-1, self.h_dim))
#         o = self.o_recon(output)
#         o = o.reshape(-1, 1, 262144)
#         o = nn.Softmax(dim=2)(o)
#         o = o.repeat(1, self.seq_len, 1)
#
#         d = self.d_recon(output)
#         d = d.reshape(-1, 1, 262144)
#         d = nn.Softmax(dim=2)(d)
#         d = d.repeat(1, self.seq_len, 1)
#
#         speed = self.speed_recon(output).reshape(-1, self.seq_len, 1)
#
#         t = self.t_recon(output)
#         t = t.reshape(-1, self.seq_len, self.seq_len)
#         t = nn.Softmax(dim=2)(t)
#         t = t.reshape(-1, self.seq_len, 1)
#
#         depart = self.depart_recon(output)
#         depart = depart.reshape(-1, 1, 144)
#         depart = nn.Softmax(dim=2)(depart)
#         depart = depart.repeat(1, self.seq_len, 1)
#
#         a = self.a_recon(output).reshape(-1, self.seq_len, 1)
#         return o, d, speed, t, depart, a
#
#
# class VAE(nn.Module):
#     def __init__(self, opt):
#         super().__init__()
#         self.z_dim = opt['z_dim']
#         self.seq_len = opt['seq_len']
#         self.obs_size = opt['obs_size']
#         self.action_size = opt['action_size']
#         self.hidden_size = opt['hidden_size']
#         self.encoder = dc_encoder(obs_size=opt['obs_size'], action_size=opt['action_size'], z_dim=opt['z_dim'])
#         self.decoder = dc_decoder(z_dim=self.z_dim, h_dim=opt['hidden_size'], obs_size=opt['obs_size'],
#                                   action_size=opt['action_size'],
#                                   seq_len=opt['seq_len'])
#         self.criterion = {"cl": nn.CrossEntropyLoss(), "mse": nn.MSELoss()}
#         self.device = opt['device']
#         self.prior_mu = torch.zeros(self.z_dim, requires_grad=False)
#         self.prior_std = torch.ones(self.z_dim, requires_grad=False)
#         self.params = list(self.parameters())
#         self.optimizer = optim.Adam(self.params, lr=opt['lr'])
#
#     def forward(self, x):
#         z_mu, z_std = self.encoder(x)
#         eps = torch.randn_like(z_mu).to(self.device)
#         z = eps.mul(z_std).add_(z_mu)
#         o, d, speed, t, depart, a = self.decoder(z)
#         kl = batch_KL_diag_gaussian_std(z_mu, z_std, self.prior_mu.to(self.device), self.prior_std.to(self.device))
#         neg_l = self.recon_loss(x, o, d, speed, t, depart, a)
#         loss = torch.mean(neg_l + kl, dim=0)
#         return loss
#
#     def recon_loss(self, x, o, d, speed, t, depart, a):
#         real_o = x[:, :, 0].reshape(-1, ).long()
#         real_d = x[:, :, 1].reshape(-1, ).long()
#         real_speed = x[:, :, 3]
#         real_t = x[:, :, 4].reshape(-1, ).long()
#         real_depart = x[:, :, 5].reshape(-1, ).long()
#         real_a = x[:, :, 6]
#         check1 = self.criterion["cl"](o.reshape(-1, o.shape[-1]), real_o)
#         check2 = self.criterion["cl"](d.reshape(-1, d.shape[-1]), real_d)
#         check3 = self.criterion["mse"](speed.squeeze(), real_speed)
#         check4 = self.criterion["cl"](t, real_t)
#         check5 = self.criterion["cl"](depart, real_depart)
#         check6 = self.criterion["cl"](a, real_a)
#
#         neg_l = check1 + check2 + check3 + check4 + check5 + check6
#         return neg_l.mean()
#
#     def sample(self, n):
#         z = torch.randn(n, self.z_dim).to(self.device)
#         return self.decoder(z)


# Encoder and Decoder using Mutual Information Estimation
class MI(nn.Module):
    def __init__(self, z_dim, device):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = encoder(z_dim=self.z_dim)
        self.device = device
        self.params = list(self.parameters())
        temp = torch.tensor(self.z_dim * [float(1) / self.z_dim])
        # To ensure more stable gradient updates for the latent skill distribution and
        # avoid issues with values that are too small or too large, we use the logarithmic
        # form of the latent skill distribution for gradient updates.
        self.prior_parameters = Variable(temp, requires_grad=True)
        self.prior_parameters = self.prior_parameters.to(self.device)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        preference = self.encoder.reparameterize(mu, log_var)
        ml = info_max_loss(self, preference)
        return preference, ml

    def train_preference(self, x, preference, value):
        sep_loss = {}
        for sample, p, v in zip(x, preference, value):
            # Extract the first two values as a tuple
            key = tuple(sample[:2].tolist())
            # Initialize the key if not present in the dictionary
            if key not in sep_loss:
                sep_loss[key] = 0.
            # Append the sample to the corresponding key
            sep_loss[key] += v * p

        reg_loss = 0
        for key, value in sep_loss.items():
            reg_loss += value

        return reg_loss


class IRL(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(384, opt['hidden_size']),
            nn.LeakyReLU(),
            nn.Linear(opt['hidden_size'], 1)
        )

    def forward(self, xp):
        r = self.net(xp)
        return r


class preference_reward_joint(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.obs_action_model = obs_action_encoder(obs_size=opt['obs_size'], h_dim=opt["hidden_size"], action_dim=opt['action_dim'] )
        self.reward_model = IRL(opt)
        if opt['reg_type'] == 'MI':
            self.preference_model = MI(opt['z_dim'], opt['device'])
        self.optimizer = optim.Adam(
            params=chain(self.reward_model.parameters(), self.preference_model.parameters()),
            lr=opt['lr']
        )

    def forward(self, x, lens):
        x = self.obs_action_model(x)
        ml = self.preference_model(x)
        p = self.preference_model.preference
        # IRL iteration
        p = p.unsqueeze(1).repeat(1, x.shape[1], 1)
        r = self.reward_model(torch.concatenate([x, p], dim=2))
        mask = torch.zeros_like(r)
        for i in range(p.shape[0]):
            mask[i, :lens[i], :] = 1
        r = r * mask
        r = r.squeeze(-1)
        r = r.sum(dim=-1)
        r = torch.exp(-r/50 + 1e-3)
        r = torch.sum(r) / torch.sum(mask)
        return r + ml * 0.1

    def get_preference(self, x):
        x = self.obs_action_model(x)
        ml = self.preference_model(x)
        return self.preference_model.preference

    def get_counterfact_reward(self, x, lens, pred_action):
        p = self.get_preference(x)
        p = p.unsqueeze(1).repeat(1, x.shape[1], 1)
        # replace the action in x with pred_action
        counter_fact_sa = torch.clone(x)
        rs = 0
        for i in range(lens.size(0)):
            start_idx = lens[i, 0]
            end_idx = lens[i, 1]
            activate = counter_fact_sa[i, start_idx:end_idx, :].unsqueeze(0)
            activate[:,:, 6] = pred_action[i, :end_idx-start_idx]
            feat = self.obs_action_model(activate)
            r = self.reward_model(torch.concatenate([feat, p[i, start_idx:end_idx, :].unsqueeze(0)], dim=2)).squeeze(-1)
            r = torch.mean(r)
            rs += torch.exp(-r/50. + 1e-3)

        return rs
