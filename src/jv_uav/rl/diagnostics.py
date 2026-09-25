"""Small episode summaries; no raw trajectories or additional random sampling."""
from __future__ import annotations
import copy
import numpy as np


def charging_record(info):
    """Distinguish a request, a reserved slot, and positive settled charge."""
    plan = info['frame_plan']
    added = np.asarray(info['settlement'].get('charge_added_frac', np.zeros(len(plan.requested_charge))))
    ids = lambda mask: np.flatnonzero(mask).tolist()
    groups = {
        'requested': ids(plan.requested_charge),
        'assigned_charging': ids(plan.assigned_status == 2),
        'assigned_waiting': ids(plan.assigned_status == 1),
        'rejected': ids(plan.requested_charge & (plan.assigned_status == 0)),
        'actually_charged': ids(added > 0),
    }
    return {'counts': {k: len(v) for k, v in groups.items()}, 'uav_ids': groups,
            'charge_added_soc_by_uav': added.tolist(),
            'settled': bool(info['settlement'])}


class EpisodeDiagnostics:
    def __init__(self, gamma):
        self.gamma = float(gamma)
        self.steps = 0
        self.return_sum = self.discounted_return = self.packets = 0.0
        self.terms = {}
        self.active_actions = self.saturated_components = self.radial_clips = self.oob_actions = 0
        self.distance_sum = self.max_radius = 0.0
        self.outer_actions = self.mid_step_deaths = self.return_failures = 0
        self.warmup_actions = 0
        self.sensor_ids = set()
        self.outer_sensor_ids = set()
        self.probes = []
        self.charging_counts = {}
        self.charging_frames = 0

    def record_charging(self, record):
        self.charging_frames += 1
        for key, value in record['counts'].items():
            self.charging_counts[key] = self.charging_counts.get(key, 0) + value

    def record(self, result, action, center, sensor_pos, *, warmup=False):
        self.return_sum += float(result.reward)
        self.discounted_return += self.gamma ** self.steps * float(result.reward)
        self.steps += 1
        for key, value in result.reward_terms.items():
            self.terms[key] = self.terms.get(key, 0.0) + float(value)
        self.packets += float(result.collected_per_sensor.sum())
        ids = result.serving_ids
        n = len(ids)
        self.active_actions += n
        self.warmup_actions += n if warmup else 0
        self.saturated_components += int((np.abs(action) > .99).sum())
        self.radial_clips += int((np.linalg.norm(action, axis=1) > 1).sum())
        self.oob_actions += int(result.oob_mask.sum())
        self.distance_sum += float(result.move_distance_m[ids].sum())
        radii = np.linalg.norm(result.position_after[ids] - center, axis=1)
        self.max_radius = max(self.max_radius, float(radii.max(initial=0)))
        self.outer_actions += int((radii > 1800).sum())
        self.mid_step_deaths += int(result.dead_during_step.size)
        self.return_failures += int(result.return_unsafe_ids.size)
        visited = np.flatnonzero(result.sensor_owner >= 0)
        self.sensor_ids.update(visited.tolist())
        self.outer_sensor_ids.update(visited[np.linalg.norm(sensor_pos[visited]-center, axis=1)>1800].tolist())

    def summary(self, episode_steps):
        n = max(self.active_actions, 1)
        probe = {}
        if self.probes:
            weights = np.array([x['components'] for x in self.probes])
            for key in ('mu_abs_mean', 'sigma_mean', 'deterministic_saturation_fraction'):
                probe[key] = float(np.average([x[key] for x in self.probes], weights=weights))
            probe.update(mu_abs_max=max(x['mu_abs_max'] for x in self.probes),
                         sigma_min=min(x['sigma_min'] for x in self.probes),
                         sigma_max=max(x['sigma_max'] for x in self.probes),
                         sampled_observations=len(self.probes))
        complete = self.steps == episode_steps
        return dict(lower_diagnostics_complete=complete, lower_recorded_steps=self.steps,
            charging_counts=dict(self.charging_counts), charging_recorded_frames=self.charging_frames,
            lower_return=self.return_sum if complete else None,
            lower_mean_reward=self.return_sum / max(self.steps, 1) if complete else None,
            lower_discounted_return=self.discounted_return if complete else None,
            lower_reward_terms_sum=dict(self.terms) if complete else None,
            lower_reward_terms_mean={k:v/max(self.steps,1) for k,v in self.terms.items()} if complete else None,
            lower_collected_packets=self.packets if complete else None,
            lower_action_summary=dict(uav_decisions=self.active_actions, warmup_uav_decisions=self.warmup_actions,
                component_saturation_fraction=self.saturated_components/(2*n),
                radial_clip_fraction=self.radial_clips/n, oob_fraction=self.oob_actions/n,
                mean_move_m=self.distance_sum/n, outer_position_fraction=self.outer_actions/n,
                max_radius_m=self.max_radius), lower_policy_probe=probe,
            lower_sensor_visited=len(self.sensor_ids), lower_outer_sensor_visited=len(self.outer_sensor_ids),
            lower_mid_step_deaths=self.mid_step_deaths, lower_return_failures=self.return_failures)

    def state_dict(self):
        return copy.deepcopy(vars(self))

    def load_state_dict(self, state):
        self.__dict__.update(copy.deepcopy(state))
