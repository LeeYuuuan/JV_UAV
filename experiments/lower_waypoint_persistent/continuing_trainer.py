"""Continuing SAC: logging windows never terminate or reset the physical task."""
from pathlib import Path
import copy
import numpy as np
import torch
from persistent_trainer import Trainer
from persistent_env import PersistentEnv
from waypoint_sampler import WaypointSampler


def summarize(env, rows, decisions):
    view = copy.copy(env)
    view.records = rows
    view.decision_records = decisions
    view.steps = len(rows)
    view.seen = {i for row in rows for i in row['covered_ids']}
    result = view.summary()
    result.update(mean_step_reward=result['lower_return'] / len(rows),
                  start_physical_step=rows[0]['step'], end_physical_step=rows[-1]['step'])
    return result


class ContinuingTrainer(Trainer):
    def __init__(self, env_cfg, cfg, device='cpu'):
        if not cfg.get('continuing_task'):
            raise ValueError('continuing experiment requires continuing_task')
        super().__init__(env_cfg, cfg, device)
        self.windows = 0
        self.window_pending = False

    def step(self):
        if self.env.done:
            raise RuntimeError('continuing environment unexpectedly terminated')
        if self.window_pending:
            # Only diagnostic buffers are cleared. No scene/reset/RNG mutation.
            self.env.records.clear()
            self.env.decision_records.clear()
            self.env.seen.clear()
            self.window_pending = False
        row = super().step()
        if row['done'] or self.replay.items[-1][4]:
            raise RuntimeError('a statistics window must never disable bootstrap')
        row.pop('episode_summary', None)
        row.pop('episode', None)
        summary = None
        end = len(self.env.records) >= self.cfg['statistics_window_steps']
        if end:
            self.windows += 1
            self.episodes = self.windows  # Serialization compatibility; not task resets.
            summary = summarize(self.env, self.env.records, self.env.decision_records)
            summary.update(window=self.windows, episode=self.windows, training_steps=self.steps,
                           sac_updates=self.updates, total_waypoint_decisions=self.decisions,
                           terminated=False, statistics_only=True,
                           behavior_counts={k: sum(r['sources'].count(k) for r in self.env.decision_records)
                                            for k in ('actor', 'random', 'greedy')})
            self.history.append(summary)
            self.window_pending = True
        return row | dict(window=self.windows if end else self.windows + 1,
                          statistics_window_end=end, window_summary=summary,
                          terminated=False, truncated=False)

    def evaluate(self, seed, output=None, mode='policy', detailed=True):
        env = PersistentEnv(self.env_cfg, self.cfg, seed=seed)
        ex = copy.deepcopy(self.cfg['exploration'])
        ex.update(epsilon_start=1., epsilon_end=1., random_probability=0.,
                  greedy_probability=1., actor_probability=0., greedy_jitter_m=0.)
        teacher = WaypointSampler(ex, self.env_cfg, seed + 20000)
        probes = []
        while env.steps < self.cfg['evaluation_steps']:
            obs = env.observation()
            if mode == 'policy':
                action = self.sac.act(obs, deterministic=True)
                probes.append(dict(physical_step=env.steps, sensor_last_visit_sec=obs['sensor_last_visit_sec'].tolist(),
                                   **self.sac.diagnostics(obs)))
            elif mode == 'greedy_reference':
                action, _ = teacher.propose(obs, np.zeros((len(obs['ids']), 2), np.float32), 0)
            else:
                raise ValueError('unknown continuing evaluation policy')
            env.step(action)
            assert not env.done
        summary = summarize(env, env.records, env.decision_records)
        summary.update(seed=seed, training_steps=self.steps, policy_mode=mode,
                       sac_updates=self.updates, terminated=False, evaluation_cutoff=True,
                       segment_summaries=[summarize(env, env.records[a:a+self.cfg.get('evaluation_page_steps',15)],
                           [d for d in env.decision_records if d['end_physical_step']>a and
                            d['start_physical_step']<a+self.cfg.get('evaluation_page_steps',15)])
                           for a in range(0, len(env.records), self.cfg.get('evaluation_page_steps',15))])
        if output is not None:
            from persistent_plots import write_json
            from continuing_plots import evaluation_pages
            output = Path(output); output.mkdir(parents=True, exist_ok=True)
            write_json(output / 'summary.json', summary)
            write_json(output / 'trace.json', dict(sensor_pos=env.scene.sensor_pos.tolist(),
                       final_sensor_backlog=env.scene.sensor_backlog.tolist(),
                       rows=env.records, decisions=env.decision_records, action_distribution=probes))
            evaluation_pages(env, output, mode=mode)
        return summary

    def save(self, path):
        state = dict(format='continuing_waypoint_sac_v1', env_cfg=self.env_cfg, cfg=self.cfg,
                     sac=self.sac.state_dict(), replay=self.replay.state_dict(), env=self.env.state_dict(),
                     sampler=self.sampler.state_dict(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None)
        for key in ('steps', 'decisions', 'updates', 'episodes', 'windows', 'window_pending',
                    'history', 'loss_history', 'last_losses'):
            state[key] = getattr(self, key)
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp'); torch.save(state, temp); temp.replace(path)

    @classmethod
    def load(cls, path, device='cpu'):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('format') != 'continuing_waypoint_sac_v1':
            raise ValueError('start a fresh continuing run; finite-horizon checkpoints contain terminal replay targets')
        obj = cls(state['env_cfg'], state['cfg'], device)
        obj.sac.load_state_dict(state['sac']); obj.replay.load_state_dict(state['replay'])
        if any(row[4] for row in obj.replay.items):
            raise ValueError('continuing replay contains terminal transitions')
        obj.env.load_state_dict(state['env']); obj.sampler.load_state_dict(state['sampler'])
        for key in ('steps', 'decisions', 'updates', 'episodes', 'windows', 'window_pending',
                    'history', 'loss_history', 'last_losses'):
            setattr(obj, key, state[key])
        torch.set_rng_state(state['torch_rng'])
        if obj.device.type == 'cuda' and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        return obj
