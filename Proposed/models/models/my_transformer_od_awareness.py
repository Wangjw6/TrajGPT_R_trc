import numpy as np
import torch
import torch.nn as nn

import transformers

from Proposed.models.models.model import TrajectoryModel
from Proposed.models.models.trajectory_gpt2 import GPT2Model
import torch.nn.functional as F


import os
import re
import pickle
def get_incremental_filename(directory, base_name, extension):
    # List all files in the directory
    files = os.listdir(directory)

    # Regular expression to match filenames in the pattern base_name_number.extension
    pattern = re.compile(rf'{base_name}_(\d+)\.{extension}')

    # Find all files that match the pattern and extract the numbers
    numbers = [int(m.group(1)) for f in files if (m := pattern.match(f))]

    # If no such files exist, start with 1
    if numbers:
        counter = max(numbers) + 1
    else:
        counter = 1

    # Construct the new file name
    file_name = f"{base_name}_{counter}.{extension}"

    return os.path.join(directory, file_name)
class My_Transformer_od_awareness(TrajectoryModel):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=False,
            normal_helper=None,
            env="toyota",
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        torch.manual_seed(1221)
        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.env = env
        self.hidden_size = hidden_size
        if env == "toyota":
            self.transformer = GPT2Model(config)
            # self.od_transformer = GPT2Model(config)
            self.normal_helper = normal_helper
            if normal_helper is not None:
                self.counts_tensor = torch.tensor([self.normal_helper[i] for i in range(len(normal_helper))],
                                                  dtype=torch.float32, device=kwargs['device'])
                self.counts_tensor = self.counts_tensor.view(1, -1)
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
            self.embed_return = torch.nn.Linear(1, hidden_size)

            self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
            self.embed_link = nn.Embedding(262144, hidden_size)
            self.embed_o = nn.Embedding(262144, hidden_size)
            self.embed_d = nn.Embedding(262144, hidden_size)
            self.embed_speed = nn.Embedding(120, hidden_size)
            self.embed_departure = nn.Embedding(144, hidden_size)

            self.embed_ln = nn.LayerNorm(hidden_size)
            self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
            self.predict_action = nn.Sequential(
                *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else [])
                  ))
            self.predict_return = torch.nn.Linear(hidden_size, 1)

            self.hidden_size = hidden_size
            self.embed_state = torch.nn.Linear(hidden_size * 5, hidden_size)
            self.is_usr = False
            if kwargs['usr'] == "all":
                self.is_usr = True
                self.embed_usr = nn.Embedding(40000, hidden_size)
                self.reasoning = nn.Linear(hidden_size, hidden_size)
        if env == "tdrive":
            self.transformer = GPT2Model(config)
            self.normal_helper = normal_helper
            if normal_helper is not None:
                self.counts_tensor = torch.tensor([self.normal_helper[i] for i in range(len(normal_helper))],
                                                  dtype=torch.float32, device=kwargs['device'])
                self.counts_tensor = self.counts_tensor.view(1, -1)
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
            self.embed_return = torch.nn.Linear(1, hidden_size)

            self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
            self.embed_link = nn.Embedding(16384, hidden_size)
            self.embed_o = nn.Embedding(16384, hidden_size)
            self.embed_d = nn.Embedding(16384, hidden_size)
            self.embed_departure = nn.Embedding(144, hidden_size)
            self.embed_state = torch.nn.Linear(hidden_size * 4, hidden_size)
            self.embed_ln = nn.LayerNorm(hidden_size)

            # note: we don't predict states or returns for the paper
            self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
            self.predict_action = nn.Sequential(
                *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else [])
                  ))
            self.predict_return = torch.nn.Linear(hidden_size, 1)

            self.hidden_size = hidden_size
            if kwargs['usr'] == "all":
                self.is_usr = True
                self.embed_usr = nn.Embedding(10000, hidden_size)
                self.reasoning = nn.Linear(hidden_size, hidden_size)

        if env == "porto":
            self.transformer = GPT2Model(config)
            self.normal_helper = normal_helper
            if normal_helper is not None:
                self.counts_tensor = torch.tensor([self.normal_helper[i] for i in range(len(normal_helper))],
                                                  dtype=torch.float32, device=kwargs['device'])
                self.counts_tensor = self.counts_tensor.view(1, -1)
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
            self.embed_return = torch.nn.Linear(1, hidden_size)

            self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
            self.embed_link = nn.Embedding(5524, hidden_size)
            self.embed_o = nn.Embedding(5524, hidden_size)
            self.embed_d = nn.Embedding(5524, hidden_size)
            self.embed_departure = nn.Embedding(144, hidden_size)
            self.embed_state = torch.nn.Linear(hidden_size * 4, hidden_size)
            self.embed_ln = nn.LayerNorm(hidden_size)

            # note: we don't predict states or returns for the paper
            self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
            self.predict_action = nn.Sequential(
                *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else [])
                  ))
            self.predict_return = torch.nn.Linear(hidden_size, 1)

            self.hidden_size = hidden_size
            if kwargs['usr'] == "all":
                self.is_usr = True
                self.embed_usr = nn.Embedding(512, hidden_size)
                self.reasoning = nn.Linear(hidden_size, hidden_size)

    def set_fixed_params(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, action_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        if self.env == "toyota":
            o, d, link, speed, depart = states[:, :, 0], states[:, :, 1], states[:, :, 2], states[:, :, 3], states[:, :, 6]
            o = o.to(torch.int64)
            d = d.to(torch.int64)
            link = link.to(torch.int64)
            speed = torch.ceil(speed).to(torch.int64)
            depart = depart.to(torch.int64)
            embed_o = self.embed_o(o)
            embed_d = self.embed_d(d)
            embed_link = self.embed_link(link)
            embed_speed = self.embed_speed(speed)
            embed_depart = self.embed_departure(depart)

            state_embeddings = torch.cat([embed_o, embed_d, embed_link, embed_speed, embed_depart], dim=-1)
            state_embeddings = self.embed_state(state_embeddings)
            if self.is_usr:
                usr_id = states[:, :, 5].to(torch.int64)
                embed_usr = self.embed_usr(usr_id)
                state_embeddings = state_embeddings + embed_usr
                preference = self.reasoning(embed_usr + embed_o + embed_d + embed_depart)

            actions = actions.to(torch.int64)
            one_hot_encoded = F.one_hot(actions, num_classes=self.act_dim)
            action_embeddings = self.embed_action(one_hot_encoded.to(torch.float32)).squeeze()
            returns_embeddings = self.embed_return(returns_to_go)
            time_embeddings = self.embed_timestep(timesteps)


        if self.env == "tdrive":
            o, d, link, depart = states[:, :, 4], states[:, :, 5], states[:, :, 0], states[:, :, 3]
            o = o.to(torch.int64)
            d = d.to(torch.int64)
            link = link.to(torch.int64)
            depart = depart.to(torch.int64)
            embed_o = self.embed_o(o)
            embed_d = self.embed_d(d)
            embed_link = self.embed_link(link)
            embed_depart = self.embed_departure(depart)
            state_embeddings = torch.cat([embed_o, embed_d, embed_link, embed_depart], dim=-1)
            state_embeddings = self.embed_state(state_embeddings)
            if self.is_usr:
                usr_id = states[:, :, 6].to(torch.int64)
                embed_usr = self.embed_usr(usr_id)
                state_embeddings = state_embeddings + embed_usr
                preference = self.reasoning(embed_usr + embed_o + embed_d + embed_depart)
            actions = actions.to(torch.int64)
            one_hot_encoded = F.one_hot(actions, num_classes=self.act_dim)
            action_embeddings = self.embed_action(one_hot_encoded.to(torch.float32)).squeeze()
            returns_embeddings = self.embed_return(returns_to_go)
            time_embeddings = self.embed_timestep(timesteps)

        if self.env == "porto":
            o, d, link, depart = states[:, :, 4], states[:, :, 5], states[:, :, 0], states[:,:,  3]
            o = o.to(torch.int64)
            d = d.to(torch.int64)
            link = link.to(torch.int64)
            depart = depart.to(torch.int64)

            embed_o = self.embed_o(o)
            embed_d = self.embed_d(d)
            embed_link = self.embed_link(link)
            embed_depart = self.embed_departure(depart)
            state_embeddings = torch.cat([embed_o, embed_d, embed_link, embed_depart], dim=-1)
            state_embeddings = self.embed_state(state_embeddings)
            if self.is_usr:
                usr_id = states[:, :, 6].to(torch.int64)
                embed_usr = self.embed_usr(usr_id)
                state_embeddings = state_embeddings + embed_usr
                preference = self.reasoning(embed_usr + embed_o + embed_d + embed_depart)
            actions = actions.to(torch.int64)  # Converting to a tensor of type Long (int64)
            one_hot_encoded = F.one_hot(actions, num_classes=self.act_dim)
            action_embeddings = self.embed_action(one_hot_encoded.to(torch.float32)).squeeze()
            returns_embeddings = self.embed_return(returns_to_go)
            time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

        transformer_outputs = self.transformer(
                inputs_embeds=stacked_inputs,
                attention_mask=stacked_attention_mask,
            )

        x = transformer_outputs['last_hidden_state']

        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:, 2])  # predict next return given state and action
        state_preds = self.predict_state(x[:, 2])  # predict next state given state and action
        action_preds = self.predict_action(x[:, 1])  # predict next action given state
        # normalize action preds
        if self.normal_helper is not None and (self.env == 'toyota' or self.env == 'porto'):
            action_preds = action_preds * sum(self.normal_helper.values())
            action_preds = action_preds / self.counts_tensor
        action_preds = F.softmax(action_preds, dim=-1)
        # TODO only return first/last two actions when using attention bootstrapping
        if self.is_usr:
            return state_preds, action_preds, preference
        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, traj_id=-1, **kwargs):
        # we don't care about the past rewards in this model
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, 1)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length - states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length - states.shape[1], self.state_dim),
                             device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], 1),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length - returns_to_go.shape[1], 1),
                             device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length - timesteps.shape[1]), device=timesteps.device),
                 timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None
        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)
        return action_preds[0, -1]
