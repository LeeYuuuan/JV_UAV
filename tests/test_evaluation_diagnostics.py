"""Evaluation must expose early failures and preserve full-episode plots."""
from unittest.mock import patch
from pathlib import Path
import json
import numpy as np

from test_rl import trainer_config
from jv_uav.rl.trainer import JointTrainer
from jv_uav.upper_env import UpperEnv


def test_evaluation_records_1000_steps_and_full_trails_with_feasible_charging():
    env, cfg = trainer_config()
    env['time']['episode_upper_frames'] = 100
    trainer = JointTrainer(env, cfg)

    def rotate(obs, deterministic=False):
        action = np.zeros(6, np.int8)
        soc = obs['actor'][0, :6]
        action[np.lexsort((np.arange(6), soc))[:4]] = 1
        return action, np.zeros(6), 0.0

    trainer.mappo.act = rotate
    trainer.sac.act = lambda obs, rms, deterministic=False: np.zeros((len(obs['active_uav_ids']), 2), np.float32)
    with patch.object(Path, 'mkdir'), patch.object(Path, 'write_text') as write, patch.object(UpperEnv, 'render') as render:
        result = trainer.evaluate(1001, 'unused-evaluation-output')
    assert result['frames'] == result['configured_frames'] == 100
    assert result['low_steps'] == result['configured_low_steps'] == 1000
    assert result['completed_horizon'] and result['termination_reason'] == 'horizon_reached'
    assert result['charge_requests'] == 400 and result['charging_assignments'] == 200
    assert len(result['charged_uav_ids']) == 6
    assert render.call_args_list[0].kwargs['trail_frames'] == 100
    policies = json.loads(write.call_args_list[2].args[0])
    assert len(policies) == 100 and len(policies[0]['request_probabilities']) == 6


def test_evaluation_reports_early_battery_death_without_extending_episode():
    env, cfg = trainer_config()
    env['time']['episode_upper_frames'] = 100
    trainer = JointTrainer(env, cfg)
    trainer.mappo.act = lambda obs, deterministic=False: (np.zeros(6, np.int8), np.zeros(6), 0.0)
    trainer.sac.act = lambda obs, rms, deterministic=False: np.zeros((len(obs['active_uav_ids']), 2), np.float32)
    result = trainer.evaluate(1001, render=False)
    assert result['configured_low_steps'] == 1000 and result['low_steps'] == 24
    assert result['termination_reason'] == 'battery_depleted_during_low_step'
    assert not result['completed_horizon']
    assert result['charge_requests'] == result['charging_assignments'] == 0
    assert result['charged_uav_ids'] == []
