from collections import deque
import copy
import random

import numpy as np
import torch
from torch import nn

from model import ActorNet, CriticNet


class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.buffer = deque(maxlen=int(capacity))

    def __len__(self):
        return len(self.buffer)

    def add(self, state, c, action, reward, next_state, next_c, done):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(c, dtype=np.float32),
                np.asarray([action], dtype=np.float32).reshape(1),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                np.asarray(next_c, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, c, a, r, ns, nc, d = zip(*batch)
        return (
            np.stack(s),
            np.stack(c),
            np.stack(a),
            np.asarray(r, dtype=np.float32)[:, None],
            np.stack(ns),
            np.stack(nc),
            np.asarray(d, dtype=np.float32)[:, None],
        )


class DDPGAgent:
    def __init__(
        self,
        state_dim=3,
        time_steps=5,
        c_dim=128,
        action_dim=1,
        action_range=40.0,
        actor_hidden=128,
        critic_hidden=128,
        num_layers=2,
        gamma=0.99,
        tau=0.001,
        actor_lr=5e-4,
        critic_lr=1e-3,
        critic_weight_decay=1e-2,
        reward_scale=1e-5,
        replay_capacity=1_000_000,
        device="cpu",
    ):
        self.time_steps = int(time_steps)
        self.c_dim = int(c_dim)
        self.action_range = float(action_range)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.reward_scale = float(reward_scale)
        self.device = torch.device(device)

        self.actor = ActorNet(
            state_dim=state_dim,
            hidden_size=actor_hidden,
            num_layers=num_layers,
            c_dim=c_dim,
            action_range=action_range,
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)

        self.critic = CriticNet(
            state_dim=state_dim,
            hidden_size=critic_hidden,
            num_layers=num_layers,
            c_dim=c_dim,
            action_dim=action_dim,
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
            weight_decay=critic_weight_decay,
        )
        self.replay = ReplayBuffer(replay_capacity)
        self.mse = nn.MSELoss()

    def load_actor_state(self, state_dict):
        self.actor.load_state_dict(state_dict)
        self.actor_target.load_state_dict(state_dict)

    @torch.no_grad()
    def act(self, state, c=None, noise_std=0.0):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_t.ndim == 2:
            state_t = state_t.unsqueeze(0)
        if c is None:
            c_t = torch.zeros((state_t.shape[0], self.c_dim), device=self.device)
        else:
            c_t = torch.as_tensor(c, dtype=torch.float32, device=self.device)
            if c_t.ndim == 1:
                c_t = c_t.unsqueeze(0)
        action = self.actor(state_t, c_t)
        if noise_std > 0:
            action = action + torch.randn_like(action) * noise_std
        action = action.clamp(-self.action_range, self.action_range)
        return float(action.squeeze().cpu())

    def add(self, state, c, action, reward, next_state, next_c, done):
        self.replay.add(state, c, action, reward, next_state, next_c, done)

    def update(self, batch_size=64):
        if len(self.replay) < batch_size:
            return None

        s, c, a, r, ns, nc, d = self.replay.sample(batch_size)
        s = torch.from_numpy(s).to(self.device)
        c = torch.from_numpy(c).to(self.device)
        a = torch.from_numpy(a).to(self.device)
        r = torch.from_numpy(r).to(self.device) * self.reward_scale
        ns = torch.from_numpy(ns).to(self.device)
        nc = torch.from_numpy(nc).to(self.device)
        d = torch.from_numpy(d).to(self.device)

        with torch.no_grad():
            next_a = self.actor_target(ns, nc)
            next_q = self.critic_target(ns, nc, next_a)
            y = r + self.gamma * (1.0 - d) * next_q

        q = self.critic(s, c, a)
        critic_loss = self.mse(q, y)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()

        pred_a = self.actor(s, c)
        actor_loss = -self.critic(s, c, pred_a).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "q_mean": float(q.detach().mean().cpu()),
            "target_q_mean": float(y.detach().mean().cpu()),
        }

    def _soft_update(self, target, source):
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.mul_(1.0 - self.tau).add_(sp, alpha=self.tau)

    def checkpoint(self, extra=None):
        payload = {
            "actor_state": {k: v.detach().cpu() for k, v in self.actor.state_dict().items()},
            "actor_target_state": {
                k: v.detach().cpu() for k, v in self.actor_target.state_dict().items()
            },
            "critic_state": {k: v.detach().cpu() for k, v in self.critic.state_dict().items()},
            "critic_target_state": {
                k: v.detach().cpu() for k, v in self.critic_target.state_dict().items()
            },
            "state_dim": 3,
            "time_steps": self.time_steps,
            "c_dim": self.c_dim,
            "hidden_size": 128,
            "num_layers": 2,
            "action_range": self.action_range,
            "gamma": self.gamma,
            "tau": self.tau,
            "reward_scale": self.reward_scale,
        }
        if extra:
            payload.update(extra)
        return payload
