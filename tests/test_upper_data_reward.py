import numpy as np
from test_core import build_env
from jv_uav.upper_env import UpperReward


def test_upper_data_mean_excludes_lower_failures_and_averages_squared_cost():
    cfg, *_ = build_env()
    reward = UpperReward(cfg)
    steps = [dict(covered_pnorm=2., system_max_post_service=-1., oob=-5., return_failure=0.),
             dict(covered_pnorm=6., system_max_post_service=-9., oob=0., return_failure=-500.)]
    value, terms = reward(6, 4000, 1, lower_return_failure_count=1, lower_reward_terms=steps)
    assert terms['frame_mean_lower_collection'] == 4
    assert terms['frame_mean_lower_backlog'] == -5
    assert value == 4 - 5 - 20 - 15
    assert 'serving' not in terms and 'frame_mean_system_max_post' not in terms
    value, terms = reward(2, 2000, 0, lower_reward_terms=steps[:1])
    assert value == 1  # Actual executed K=1, not configured frame length=10.


def test_upper_data_reward_matches_live_lower_terms_full_and_terminal_frame():
    for early_death in (False, True):
        _, scene, _, _, env = build_env()
        if early_death:
            scene.uav_battery[0] = 0.001
        _, value, terminated, _, info = env.step(np.zeros(6, dtype=np.int8))
        steps = info['low_frame'].steps
        assert len(steps) == (1 if early_death else 10)
        assert terminated == early_death
        expected = np.mean([s.reward_terms['covered_pnorm'] +
                            s.reward_terms['system_max_post_service'] for s in steps])
        expected -= 20 if early_death else 0
        assert np.isclose(value, expected)
        assert np.isclose(value, sum(info['reward_terms'].values()))
