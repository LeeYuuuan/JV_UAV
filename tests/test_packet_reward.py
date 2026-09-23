"""Actual packet rewards are additive and count overlapped service only once."""
from types import SimpleNamespace
import numpy as np
from test_core import build_env
from jv_uav.models import FullClearNearestService
from jv_uav.low_env import LowReward


def test_packet_collection_not_double_counted_by_overlapping_uavs():
    cfg,*_=build_env()
    service=FullClearNearestService(6,173.2)
    sensors=np.array([[0.,0.],[200.,0.]])
    positions=np.full((6,2),4000.);positions[:2]=[100,0]
    packets=np.array([600.,600.])
    result=service.collect(sensors,packets,positions,np.array([0,1]))
    def terms(result):
        return LowReward(cfg)(SimpleNamespace(collected_per_sensor=result.collected_per_sensor,
            per_uav_owned_max_pre_service=result.per_uav_max,system_max_post_service=0.,oob_mask=np.zeros(6,bool)))[1]
    assert result.collected_per_sensor.sum()==1200
    assert terms(result)['collected_packets']==200
    assert 'covered_max_sum' not in terms(result)
    repeated=service.collect(sensors,packets-result.collected_per_sensor,positions,np.array([0,1]))
    assert terms(repeated)['collected_packets']==0


def test_packet_reward_removes_incentive_to_split_the_same_collection():
    cfg,*_=build_env()
    service=FullClearNearestService(6,173.2);reward=LowReward(cfg)
    sensors=np.array([[0.,0.],[200.,0.],[3000.,3000.]])
    totals=[]
    for path in [[[100.,0.],[100.,0.]], [[0.,0.],[180.,0.]]]:
        packets=np.array([600.,600.,10000.]);rewards=[];positive=0
        for step,pos in enumerate(path):
            if step:packets+=30
            positions=np.full((6,2),4000.);positions[0]=pos
            collected=service.collect(sensors,packets,positions,np.array([0,1]))
            packets-=collected.collected_per_sensor
            value,terms=reward(SimpleNamespace(collected_per_sensor=collected.collected_per_sensor,
                per_uav_owned_max_pre_service=collected.per_uav_max,system_max_post_service=packets.max(),oob_mask=np.zeros(6,bool)))
            rewards.append(value);positive+=terms['collected_packets']
        totals.append((sum(rewards),rewards[0]+.99*rewards[1],positive))
    assert totals[0][2]==210 and totals[1][2]==205
    assert totals[0][0]>totals[1][0] and totals[0][1]>totals[1][1]
