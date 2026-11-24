"""Set-based SAC: serving queries, non-serving context, age-only sensor background."""
from __future__ import annotations
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from jv_uav.rl.sac import backward_step


class Encoder(nn.Module):
    def __init__(self, sensors, hidden, heads, critic=False):
        super().__init__()
        self.critic = critic
        self.uav = nn.Sequential(nn.Linear(10, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.target = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.age = nn.Sequential(nn.Linear(sensors, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.phase = nn.Linear(1, hidden)
        self.null = nn.Parameter(torch.zeros(1, 1, hidden))
        self.team = nn.MultiheadAttention(hidden, heads, dropout=0, batch_first=True)
        self.other = nn.MultiheadAttention(hidden, heads, dropout=0, batch_first=True)
        self.sensor = nn.MultiheadAttention(hidden, heads, dropout=0, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])
        self.action = nn.Linear(2, hidden) if critic else None
        self.fuse = nn.Sequential(nn.Linear(hidden*2, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def forward(self, batch, actions=None):
        fleet = self.uav(batch['uav'])
        if self.critic:
            query = fleet + self.action(actions * batch['active'][..., None])
            valid = torch.ones_like(batch['active'])
        else:
            query = torch.gather(fleet, 1, batch['query_ids'][..., None].expand(-1, -1, fleet.shape[-1]))
            valid = batch['query_mask']
        # A dummy query prevents all-masked attention in zero-service transitions.
        safe_valid = valid.clone()
        safe_valid[~safe_valid.any(1), 0] = True
        h, _ = self.team(query, query, query, key_padding_mask=~safe_valid, need_weights=False)
        query = self.norms[0](query+h)
        context = torch.cat((fleet, self.null.expand(len(fleet), -1, -1)), 1)
        context_mask = torch.cat((batch['active'], torch.zeros((len(fleet), 1), device=fleet.device, dtype=torch.bool)), 1)
        h, _ = self.other(query, context, context, key_padding_mask=context_mask, need_weights=False)
        query = self.norms[1](query+h)
        target = self.target(batch['targets'])
        h, _ = self.sensor(query, target, target, need_weights=False)
        query = self.norms[2](query+h)
        global_h = self.age(batch['age']) + self.phase(batch['phase'])
        query = self.fuse(torch.cat((query, global_h[:, None].expand_as(query)), -1))
        return query, valid


class Actor(nn.Module):
    def __init__(self, sensors, hidden, heads):
        super().__init__()
        self.encoder = Encoder(sensors, hidden, heads)
        self.mean = nn.Linear(hidden, 2)
        self.log_std = nn.Linear(hidden, 2)

    def distribution(self, batch):
        h, mask = self.encoder(batch)
        return self.mean(h), self.log_std(h).clamp(-5, 2), mask

    def forward(self, batch, deterministic=False):
        mean, log_std, mask = self.distribution(batch)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        z = mean if deterministic else distribution.rsample()
        query_actions = z.tanh()*mask[..., None]
        logp = (distribution.log_prob(z)-2*(math.log(2)-z-F.softplus(-2*z))).sum(-1)
        # Scatter-add is safe even for duplicated dummy ID 0: its action is zero.
        actions = torch.zeros((*batch['active'].shape, 2), device=z.device)
        actions = actions.scatter_add(1, batch['query_ids'][..., None].expand_as(query_actions), query_actions)
        return actions, (logp*mask).sum(-1)


class Q(nn.Module):
    def __init__(self, sensors, hidden, heads):
        super().__init__()
        self.encoder = Encoder(sensors, hidden, heads, critic=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, batch, actions):
        h, _ = self.encoder(batch, actions)
        return self.head(h.mean(1)).squeeze(-1)


class TwinQ(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.q1, self.q2 = Q(*args), Q(*args)

    def forward(self, batch, actions):
        return self.q1(batch, actions), self.q2(batch, actions)


class SAC:
    def __init__(self, env, cfg, device):
        self.device, self.cfg = torch.device(device), copy.deepcopy(cfg)
        self.size, self.age_scale, self.fleet = env.size, env.cfg['age_scale_sec'], env.n
        args = (env.m, cfg['hidden'], cfg['heads'])
        self.actor = Actor(*args).to(self.device)
        self.critic = TwinQ(*args).to(self.device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg['lr'])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg['lr'])
        self.log_alpha = torch.tensor(math.log(cfg['alpha']), device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg['lr'])

    def pack(self, observations):
        count = len(observations)
        active = np.zeros((count, self.fleet), bool)
        qcount = max(1, max(len(o['ids']) for o in observations))
        query_ids = np.zeros((count, qcount), np.int64)
        query_mask = np.zeros((count, qcount), bool)
        uav = np.zeros((count, self.fleet, 10), np.float32)
        for b, obs in enumerate(observations):
            ids = obs['ids']
            active[b, ids] = True
            query_ids[b, :len(ids)] = ids
            query_mask[b, :len(ids)] = True
            uav[b, :, :2] = obs['xy']/self.size
            uav[b, :, 2] = obs['soc']
            uav[b, :, 3] = np.arange(self.fleet)/max(self.fleet-1, 1)
            uav[b, :, 4:] = np.eye(6)[obs['status']]
        targets = np.stack([np.column_stack((o['targets']/self.size, o['target_age']/self.age_scale)) for o in observations])
        values = dict(active=active, query_ids=query_ids, query_mask=query_mask, uav=uav,
                      targets=targets.astype(np.float32), age=np.stack([o['age']/self.age_scale for o in observations]),
                      phase=np.array([[o['phase']] for o in observations], dtype=np.float32))
        return {k:torch.as_tensor(v, device=self.device) for k,v in values.items()}

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        if not len(obs['ids']):
            return np.zeros((0, 2), np.float32)
        actions, _ = self.actor(self.pack([obs]), deterministic)
        return actions[0, obs['ids']].cpu().numpy()

    @torch.no_grad()
    def diagnostics(self, obs):
        mean, log_std, mask = self.actor.distribution(self.pack([obs]))
        valid = mask[0]
        return dict(ids=obs['ids'].tolist(), mu=mean[0, valid].cpu().tolist(),
                    sigma=log_std[0, valid].exp().cpu().tolist(), mean_action=mean[0, valid].tanh().cpu().tolist())

    def update(self, transitions):
        obs, actions, rewards, future, terminated = zip(*transitions)
        batch, nxt = self.pack(obs), self.pack(future)
        padded = np.zeros((len(obs), self.fleet, 2), np.float32)
        for i, o in enumerate(obs):
            padded[i, o['ids']] = actions[i]
        action = torch.as_tensor(padded, device=self.device)
        reward = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        end = torch.as_tensor(terminated, device=self.device, dtype=torch.float32)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a, logp = self.actor(nxt)
            q1, q2 = self.target(nxt, a)
            target = reward + self.cfg['gamma']*(1-end)*(torch.minimum(q1, q2)-alpha*logp)
        q1, q2 = self.critic(batch, action)
        loss1, loss2 = F.smooth_l1_loss(q1, target), F.smooth_l1_loss(q2, target)
        critic_loss = loss1+loss2
        backward_step(critic_loss, self.critic_opt, self.critic.parameters(), self.cfg['max_grad_norm'])
        metrics = dict(critic_loss=float(critic_loss.detach()), q1_loss=float(loss1.detach()),
                       q2_loss=float(loss2.detach()), target_mean=float(target.mean()),
                       q_mean=float(torch.minimum(q1, q2).detach().mean()), reward_mean=float(reward.mean()),
                       actor_loss=0., policy_loss=0., entropy_loss=0., alpha_loss=0.,
                       mean_logp=0., action_saturation=0., sigma_mean=0.)
        eligible = batch['active'].any(1)
        if eligible.any():
            selected = {k:v[eligible] for k,v in batch.items()}
            self.critic.requires_grad_(False)
            try:
                a, logp = self.actor(selected)
                qa, qb = self.critic(selected, a)
                policy_loss = -torch.minimum(qa, qb).mean()
                entropy_loss = (alpha*logp).mean()
                actor_loss = policy_loss+entropy_loss
                backward_step(actor_loss, self.actor_opt, self.actor.parameters(), self.cfg['max_grad_norm'])
            finally:
                self.critic.requires_grad_(True)
            if self.cfg['auto_alpha']:
                target_entropy = -2*selected['active'].sum(1)
                alpha_loss = -(self.log_alpha*(logp.detach()+target_entropy)).mean()
                backward_step(alpha_loss, self.alpha_opt, [self.log_alpha], self.cfg['max_grad_norm'])
                with torch.no_grad():
                    self.log_alpha.clamp_(-12, 3)
                metrics['alpha_loss'] = float(alpha_loss.detach())
            with torch.no_grad():
                _, log_std, mask = self.actor.distribution(selected)
                metrics['sigma_mean'] = float(log_std[mask].exp().mean())
                metrics['action_saturation'] = float((a[selected['active']].abs()>.99).float().mean())
            metrics.update(actor_loss=float(actor_loss.detach()), policy_loss=float(policy_loss.detach()),
                           entropy_loss=float(entropy_loss.detach()), mean_logp=float(logp.detach().mean()))
        with torch.no_grad():
            for p, t in zip(self.critic.parameters(), self.target.parameters()):
                t.lerp_(p, self.cfg['tau'])
        metrics.update(alpha=float(self.log_alpha.exp().detach()),
                       input_age_max=float(batch['age'].max()), active_mean=float(batch['active'].float().sum(1).mean()))
        return metrics

    def state_dict(self):
        return {k:getattr(self, k).state_dict() for k in
                ('actor','critic','target','actor_opt','critic_opt','alpha_opt')} | {'log_alpha':self.log_alpha.detach().cpu()}

    def load_state_dict(self, state):
        for k in ('actor','critic','target','actor_opt','critic_opt','alpha_opt'):
            getattr(self, k).load_state_dict(state[k])
        with torch.no_grad():
            self.log_alpha.copy_(state['log_alpha'].to(self.device))


class Replay:
    def __init__(self, capacity, seed):
        self.capacity, self.index, self.items = capacity, 0, []
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, nxt, terminated):
        item = (copy.deepcopy(obs), action.copy(), float(reward), copy.deepcopy(nxt), bool(terminated))
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            self.items[self.index] = item
        self.index = (self.index+1)%self.capacity

    def sample(self, count):
        return [self.items[i] for i in self.rng.choice(len(self.items), count, replace=False)]
