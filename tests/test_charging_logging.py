import numpy as np
from test_core import build_env
from jv_uav.rl.diagnostics import charging_record, EpisodeDiagnostics


def test_requests_partition_and_unsettled_frame_never_counts_as_charged():
    _, scene, _, _, env = build_env()
    # Six requests compete for four zone positions. One rejected UAV dies early.
    scene.uav_battery.fill(.001)
    _, _, terminated, _, info = env.step(np.ones(6, dtype=np.int8))
    record = charging_record(info)
    assert terminated and not record['settled']
    assert record['counts'] == dict(requested=6, assigned_charging=2,
                                   assigned_waiting=2, rejected=2, actually_charged=0)
    assert sum(record['charge_added_soc_by_uav']) == 0


def test_actual_charge_and_episode_checkpoint_accounting():
    _, scene, _, _, env = build_env()
    scene.uav_battery.fill(.6)
    _, _, _, _, info = env.step(np.ones(6, dtype=np.int8))
    record = charging_record(info)
    assert record['settled'] and record['counts']['actually_charged'] == 2
    diagnostics = EpisodeDiagnostics(.99)
    diagnostics.record_charging(record)
    restored = EpisodeDiagnostics(.99)
    restored.load_state_dict(diagnostics.state_dict())
    restored.record_charging(record)
    assert restored.charging_frames == 2
    assert restored.charging_counts['requested'] == 12
    assert restored.charging_counts['actually_charged'] == 4
