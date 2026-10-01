from types import SimpleNamespace
import numpy as np
from test_core import build_env
from jv_uav.low_env import LowReward
from jv_uav.models import FullClearNearestService


def test_union_pnorm_counts_overlap_once_and_keeps_backlog_cost():
    cfg, *_ = build_env()
    model = FullClearNearestService(6, 173.2)
    sensors = np.array([[0., 0.], [200., 0.], [3000., 3000.]])
    uavs = np.full((6, 2), 4000.); uavs[:2] = [100., 0.]
    collected = model.collect(sensors, np.array([600., 600., 6000.]), uavs, np.array([0, 1]))
    result = SimpleNamespace(collected_per_sensor=collected.collected_per_sensor,
                             system_max_post_service=6000., oob_mask=np.zeros(6, bool))
    value, terms = LowReward(cfg)(result)
    assert np.isclose(terms['covered_pnorm'], 600 * 2**.25 / 180)
    assert terms['system_max_post_service'] == -9
    assert np.isclose(value, 600 * 2**.25 / 180 - 9)
    result.collected_per_sensor[:] = 0
    assert LowReward(cfg)(result)[1]['covered_pnorm'] == 0


def test_pnorm_rewards_high_backlogs_and_additional_useful_sensors():
    cfg, *_ = build_env()
    def positive(packets):
        r = SimpleNamespace(collected_per_sensor=np.array(packets, dtype=float),
                            system_max_post_service=0., oob_mask=np.zeros(6, bool))
        return LowReward(cfg)(r)[1]['covered_pnorm']
    assert positive([100., 80.]) > positive([100., 10., 10., 10., 10.]) > positive([100.])
    assert np.isclose(positive([100., 80.]), positive([80., 100.]))
    assert np.isfinite(positive([1e100, 1e100]))
    cfg['reward']['low_collection_p'] = 1
    assert np.isclose(positive([100., 80.]), 1)
