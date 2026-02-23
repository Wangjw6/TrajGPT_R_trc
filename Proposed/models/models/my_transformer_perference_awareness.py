import numpy as np
import transformers

from Proposed.models.models.model import TrajectoryModel
from Proposed.models.models.trajectory_gpt2 import GPT2Model
from Proposed.models.models.ft_loss_baseline import *
import torch.nn.functional as F


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


class My_Transformer_preference_awareness(TrajectoryModel):
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
            # preference_model=None,
            env="toyota",
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.env = env
        torch.manual_seed(6144)
        self.hidden_size = hidden_size
        self.device = kwargs['device']
        self.ft_type = kwargs['ft_type']
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.is_usr = False
        if kwargs['usr'] == "all":
            self.is_usr = True
        if env == "toyota":
            self.transformer = GPT2Model(config)
            self.od_transformer = GPT2Model(config)
            self.normal_helper = normal_helper
            self.N = sum(self.normal_helper.values())
            self.counts_tensor = torch.tensor([self.normal_helper[i] for i in range(len(normal_helper))],
                                              dtype=torch.float32, device=kwargs['device'])
            self.counts_tensor = self.counts_tensor.view(1, -1)
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
            self.embed_return = torch.nn.Linear(1, hidden_size)

            self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

            self.embed_link = nn.Embedding(262144, hidden_size)
            self.embed_o = nn.Embedding(262144, hidden_size)
            self.embed_d = nn.Embedding(262144, hidden_size)
            self.embed_speed = nn.Embedding(120, hidden_size)  # torch.nn.Linear(1, hidden_size)
            self.embed_departure = nn.Embedding(144, hidden_size)
            self.embed_state = torch.nn.Linear(hidden_size * 5, hidden_size)
            self.embed_ln = nn.LayerNorm(hidden_size)

            # note: we don't predict states or returns for the paper
            self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
            self.predict_action = nn.Sequential(
                *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else [])
                  ))
            if kwargs['usr'] == "all":
                self.is_usr = True
                self.embed_usr = nn.Embedding(40000, hidden_size)
                self.reasoning = nn.Linear(hidden_size, hidden_size)
            self.predict_return = torch.nn.Linear(hidden_size, 1)
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
            if kwargs['usr'] == "all":
                self.is_usr = True
                self.embed_usr = nn.Embedding(10000, hidden_size)
                self.reasoning = nn.Linear(hidden_size, hidden_size)
            self.hidden_size = hidden_size

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
            # self.preference_token_embedding = torch.zeros(1, 1, self.hidden_size)
            if kwargs['usr'] == "all":
                self.is_usr = True
                self.embed_usr = nn.Embedding(512, hidden_size)
                self.reasoning = nn.Linear(hidden_size, hidden_size)

        self.ref_model = None
        # self.preference_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.adaptive = False
        self.alpha = torch.tensor(0.01, requires_grad=False, device=kwargs['device'])  # .exp()

        self.scalar_head = layer_init(
            nn.Linear(self.hidden_size, 1),
            std=1 / np.sqrt(self.hidden_size + 1),
        )

        self.hyperparameters = {
                                'toyota': {'kl_param': 0.1, 'rm_tau': 0.7},
                                'tdrive': {'kl_param': 0.3, 'rm_tau': 0.8},
                                'porto': {'kl_param': 0.3, 'rm_tau': 0.8}
                                }

    def set_preference_model_fix(self, preference_model=None):
        self.preference_model = preference_model
        for param in self.preference_model.parameters():
            param.requires_grad = False

        # Unfreeze the parameters of embed_state
        for param in self.preference_model.preference_token_embedding.parameters():
            param.requires_grad = True

    def train_preference_mode(self, ):
        # freeze all the embedding layers
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.adaptive = True

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, action_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        q_1, preference, q_2 = self.preference_model(states.reshape(-1, states.shape[-1]), None)
        q = q_1 + q_2
        q = q.detach()
        if self.adaptive:
            v = self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True).reshape(batch_size, seq_length,
                                                                                          1)
        else:
            v = self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True).reshape(batch_size, seq_length,
                                                                                          1).detach()
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        if self.env == "toyota":
            r = torch.clip(v / 100., -1., 1.)
            returns_to_go = torch.where(returns_to_go == 0, returns_to_go, r)
            o, d, link, speed, depart = states[:, :, 0], states[:, :, 1], states[:, :, 2], states[:, :, 3], states[:, :,
                                                                                                            6]
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
            actions = actions.to(torch.int64)  # Converting to a tensor of type Long (int64)
            one_hot_encoded = F.one_hot(actions, num_classes=self.act_dim)
            action_embeddings = self.embed_action(one_hot_encoded.to(torch.float32)).squeeze()
            returns_embeddings = self.embed_return(returns_to_go)
            time_embeddings = self.embed_timestep(timesteps)

        if self.env == "tdrive":
            r = torch.clip(v / 100., -1., 1.)
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
            actions = actions.to(torch.int64)  # Converting to a tensor of type Long (int64)
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

        # time embeddings are treated similar to positional embeddings
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
        att = transformer_outputs['cross_attentions']
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:, 2])  # predict next return given state and action
        state_preds = self.predict_state(x[:, 2])  # predict next state given state and action
        action_preds = self.predict_action(x[:, 1])  # predict next action given state
        # normalize action preds
        if self.normal_helper is not None and self.env == 'toyota':
            action_preds = action_preds * sum(self.normal_helper.values())
            action_preds = action_preds / self.counts_tensor

        action_preds = F.softmax(action_preds, dim=-1)
        # concat action and hidden state
        query_response = x[:, 1] + x[:, 2]
        if self.is_usr:
            return state_preds, action_preds, [att, q_1 + q_2, query_response]
        return state_preds, action_preds, [query_response, q_1 + q_2]


    def preference_reg2(self, obs, full_trajs, lens, pred_action, attention_mask, last_hidden, old_pred_action=None):
        batch_size, num_obs, num_classes = pred_action.shape
        full_trajs_repeated = full_trajs.repeat_interleave(obs.shape[1], dim=0)
        value_pred = self.scalar_head(last_hidden.detach()).squeeze(-1)
        action_preds = pred_action.view(batch_size, num_obs, num_classes)
        action_preds = action_preds / 0.7
        old_pred_action = old_pred_action.view(batch_size, num_obs, num_classes)
        max_indices = torch.argmax(action_preds, dim=-1)
        log_probs = F.log_softmax(action_preds, dim=-1)
        log_probs = log_probs.gather(dim=-1, index=max_indices.unsqueeze(-1))
        old_pred_action = old_pred_action / 0.7
        log_probs_old = F.log_softmax(old_pred_action, dim=-1)
        log_probs_old = log_probs_old.gather(dim=-1, index=max_indices.unsqueeze(-1)).detach()
        kl = log_probs.squeeze(-1) - log_probs_old.squeeze(-1)
        with torch.no_grad():
            q, preference, q_2 = self.get_full_q(obs.view(-1, obs.shape[-1]), [full_trajs_repeated, lens])
            q = q.view(batch_size, num_obs, num_classes)
            v = self.alpha * torch.logsumexp(q / self.alpha, dim=-1, keepdim=True).squeeze(-1)
            action_index = torch.argmax(pred_action, dim=-1)
            q_a = q.gather(2, action_index.unsqueeze(-1)).squeeze(-1)
            score = q_a - v
            r = -kl * 0.15
            r[:, -1] += score[:, -1]
            r = self.whiten(r, shift_mean=True)
            gamma = 1.
            returns_reversed = []
            for i in reversed(range(num_obs)):
                if i == num_obs - 1:
                    returns_reversed.append(value_pred[:, i])
                else:
                    returns_reversed.append(r[:, i] + gamma * returns_reversed[-1])
            returns = torch.stack(returns_reversed[::-1], axis=1)
            advantages_reversed = []
            lastgaelam = 0.
            for t in reversed(range(num_obs)):
                nextvalues = value_pred[:, t + 1] if t < num_obs - 1 else 0.0
                delta = r[:, t] + gamma * nextvalues - value_pred[:, t]
                lastgaelam = delta + gamma * 0.95 * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], axis=1)
        v_loss = F.mse_loss(value_pred, returns)
        ratios = torch.exp(log_probs - log_probs_old).reshape(-1)[attention_mask.reshape(-1) > 0]
        advantages = advantages.reshape(-1)[attention_mask.reshape(-1) > 0]
        advantages = self.whiten(advantages, shift_mean=True)
        surr1 = ratios * advantages.reshape(-1)
        eps_clip = 0.2
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages.reshape(-1)
        policy_loss = -torch.min(surr1, surr2) + 0.1 * v_loss
        return policy_loss.mean()  # .detach()

    def finetune_loss(self, obs, full_trajs, lens, pred_action, attention_mask, action_target, old_pred_action=None,
                      aux=None, update_steps=0, dones=None, **kwargs):
        batch_size, num_obs, num_classes = pred_action.shape
        full_trajs_repeated = full_trajs.repeat_interleave(obs.shape[1], dim=0)
        q, preference, q_2 = self.get_full_q(obs.view(-1, obs.shape[-1]), [full_trajs_repeated, lens], ref=False)
        q = q.view(batch_size, num_obs, num_classes)
        action_target = action_target.reshape(-1, 1)
        real_indices = action_target.to(torch.int64).squeeze(-1)
        action_preds = pred_action.view(batch_size, num_obs, num_classes)
        max_indices = torch.argmax(action_preds, dim=-1).reshape(-1)

        poor_indices = torch.argmax(q, dim=-1).reshape(-1)

        old_pred_action = old_pred_action.view(batch_size, num_obs, num_classes)

        q_masked = q.reshape(-1, num_classes)
        q_masked = F.log_softmax(q_masked, dim=-1)
        q_target = q_masked[torch.arange(q_masked.size(0)), real_indices][attention_mask.reshape(-1) > 0]
        q_selected = q_masked[torch.arange(q_masked.size(0)), max_indices][attention_mask.reshape(-1) > 0]
        irl_loss = (torch.sigmoid(q_selected - q_target)).mean()  # fine-tuning the irl based reward model online

        action_preds = action_preds / 0.7
        log_probs = F.log_softmax(action_preds, dim=-1).reshape(-1, num_classes)
        policy_chosen_logps = log_probs.gather(dim=-1, index=real_indices.unsqueeze(-1))
        policy_rejected_logps1 = log_probs.gather(dim=-1, index=max_indices.unsqueeze(-1))
        policy_rejected_logps2 = log_probs.gather(dim=-1, index=poor_indices.unsqueeze(-1))

        old_pred_action = old_pred_action / 0.7
        log_probs_old = F.log_softmax(old_pred_action, dim=-1).reshape(-1, num_classes)
        reference_chosen_logps = log_probs_old.gather(dim=-1, index=real_indices.unsqueeze(-1)).detach()
        reference_rejected_logps1 = log_probs_old.gather(dim=-1, index=max_indices.unsqueeze(-1)).detach()
        reference_rejected_logps2 = log_probs_old.gather(dim=-1, index=poor_indices.unsqueeze(-1)).detach()

        if self.ft_type == "dpo":
            losses, chosen_rewards, rejected_rewards = dpo_loss(policy_chosen_logps, policy_rejected_logps1,
                                                                reference_chosen_logps, reference_rejected_logps1,
                                                                beta=0.3)
            return losses.mean()

        if aux is not None:
            preference_train = aux[0]
            last_hidden = aux[2]

        if "rf*" in self.ft_type:
            w = float(self.ft_type.split("*")[1]) / 100.
            losses1, chosen_rewards, rejected_rewards = dpo_loss(policy_chosen_logps, policy_rejected_logps1,
                                                                 reference_chosen_logps, reference_rejected_logps1,
                                                                 beta=0.3)
            losses2, chosen_rewards, rejected_rewards = dpo_loss(policy_chosen_logps, policy_rejected_logps2,
                                                                 reference_chosen_logps, reference_rejected_logps2,
                                                                 beta=0.3)
            a = 0.8
            return a * losses1.mean() + (1 - a) * losses2.mean() + w * irl_loss.mean()

        if "rlhfw" in self.ft_type:
            eps_clip = 0.2
            w = float(self.ft_type.split("w")[1]) / 100.

            last_hidden_normalization = self.observation_normalization(last_hidden.detach())
            value_pred = self.scalar_head(last_hidden_normalization).squeeze(-1)
            with torch.no_grad():
                q, preference, q_2 = self.get_full_q(obs.view(-1, obs.shape[-1]), [full_trajs_repeated, lens], ref=True)
                q = q.view(batch_size, num_obs, num_classes)
                v = self.alpha * torch.logsumexp(q / self.alpha, dim=-1, keepdim=True).squeeze(-1)
                action_index = torch.argmax(pred_action, dim=-1)
                q_a = q.gather(2, action_index.unsqueeze(-1)).squeeze(-1)
                score = q_a - v * 0.9 * (1 - dones)
                kl = policy_chosen_logps.view(batch_size, num_obs) - reference_chosen_logps.view(batch_size, num_obs)
                r = -kl * self.hyperparameters[self.env]['kl_param']
                r += score
                batch_size = lens.shape[0]
                gamma = 1.
                returns_reversed = []
                for i in reversed(range(num_obs)):
                    if i == num_obs - 1:
                        returns_reversed.append(value_pred[:, i])
                    else:
                        returns_reversed.append(r[:, i] + gamma * returns_reversed[-1])
                returns = torch.stack(returns_reversed[::-1], axis=1)
                returns = self.whiten(returns, shift_mean=True)
                advantages_reversed = []
                lastgaelam = 0.
                for t in reversed(range(num_obs)):
                    nextvalues = value_pred[:, t + 1] if t < num_obs - 1 else 0.0
                    delta = r[:, t] + gamma * nextvalues - value_pred[:, t]
                    lastgaelam = delta + gamma * 0.95 * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
            value_pred_clipped = torch.clip(value_pred, value_pred - eps_clip, value_pred + eps_clip)
            v_loss_clipped = F.mse_loss(value_pred_clipped.reshape(-1)[attention_mask.reshape(-1) > 0],
                                        returns.reshape(-1)[attention_mask.reshape(-1) > 0])
            v_loss_unclipped = F.mse_loss(value_pred.reshape(-1)[attention_mask.reshape(-1) > 0],
                                returns.reshape(-1)[attention_mask.reshape(-1) > 0])
            v_loss = torch.max(v_loss_clipped, v_loss_unclipped)
            ratios = torch.exp(policy_rejected_logps1.view(batch_size, num_obs) - reference_rejected_logps1.view(batch_size, num_obs))[:, -1].reshape(-1)
            advantages = advantages[:, -1].reshape(-1)
            surr1 = ratios * advantages.reshape(-1)

            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages.reshape(-1)

            policy_loss = -torch.min(surr1, surr2).mean() + v_loss
        return policy_loss


    def whiten(self, values, shift_mean=True):
        mean = torch.mean(values, dim=tuple(range(values.dim())), keepdim=True)
        var = torch.var(values, dim=tuple(range(values.dim())), unbiased=False, keepdim=True)

        whitened = (values - mean) / torch.sqrt(var + 1e-8)

        if not shift_mean:
            whitened += mean

        return whitened

    def observation_normalization(self, observations):
        eps = 1e-8
        mean = observations.mean(dim=0, keepdim=True)
        std = observations.std(dim=0, keepdim=True)
        normalized_obs = (observations - mean) / (std + eps)
        return torch.clamp(normalized_obs, -10., 10.)

    def get_full_q(self, obs, traj=None, ref=False):
        if traj is not None:
            if ref:
                q_1, preference, q_2 = self.ref_model.preference_model(obs, traj)
            else:
                q_1, preference, q_2 = self.preference_model(obs, traj)
            if q_2 is not None:
                current_Q = q_1 + q_2
            else:
                current_Q = q_1
            return current_Q, preference, q_2
        q = self.q_net(obs)
        return q, None, None


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


