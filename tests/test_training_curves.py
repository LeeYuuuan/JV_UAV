"""Only completed episodes enter curves; resume excludes discarded future data."""
import io
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np

from test_rl import trainer_config
from jv_uav.rl.trainer import JointTrainer
from jv_uav.rl.curves import TrainingCurves


def test_curves_use_all_episode_low_steps_excluding_cold_start():
    env, cfg = trainer_config()
    env['time']['episode_upper_frames'] = 1
    trainer = JointTrainer(env, cfg)
    trainer.mappo.act = lambda obs: (np.zeros(6, np.int8), np.zeros(6), 0.0)
    row = trainer.step()
    values = trainer.env.trace.backlog_max_post_timeline[1:]
    assert row['episode_metrics']['mean_max_backlog'] == np.mean(values)
    assert row['episode_metrics']['peak_max_backlog'] == max(values)
    assert row['episode_metrics']['low_steps'] == len(values)


def test_curve_history_rewinds_and_does_not_fabricate_legacy_backlog():
    def row(ep, step):
        return dict(episodes=ep, upper_steps=step, episode_return=-ep,
                    terminated=True, truncated=False, dead_ids=[0])
    history = ''.join(json.dumps(row(ep, ep * 10)) + '\n' for ep in [1, 2, 3]) + '{partial'
    with patch.object(Path, 'exists', lambda p: p.name == 'train.jsonl'), patch.object(Path, 'open', lambda *a, **k: io.StringIO(history)):
        curves = TrainingCurves('unused', upper_steps=20)
    assert list(curves.episodes) == [1, 2]
    assert 'mean_max_backlog' not in curves.episodes[1]
    partial = row(2, 21)
    partial.update(terminated=False, truncated=False)
    curves.add(partial)
    assert curves.episodes[2]['upper_steps'] == 20
    newer = row(2, 19)
    newer['episode_metrics'] = dict(mean_max_backlog=123, peak_max_backlog=200, low_steps=15)
    curves.add(newer)
    assert curves.episodes[2]['mean_max_backlog'] == 123
    assert len(curves.episodes) == 2
