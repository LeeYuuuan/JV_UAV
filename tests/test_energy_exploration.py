"""Energy-aware guided exploration must leave random overrides unconstrained."""
import numpy as np
from test_core import build_env
from jv_uav.rl.exploration import WaypointExploration


def test_guided_action_turns_inward_before_frame_end_failure():
    _,scene,energy,*_=build_env()
    scene.step_in_frame=9
    scene.uav_pos[0]=[3800,2000]
    scene.uav_battery[0]=.23
    obs=scene.observe_lower()
    obs['active_uav_ids']=obs['active_uav_ids'][:1]
    obs['active_uav_features']=obs['active_uav_features'][:1]
    explorer=WaypointExploration(scene,np.random.default_rng(123),0,energy_guard=True,energy_model=energy)
    explorer.valid[0]=True
    explorer.targets[0]=[3990,2000]
    action=explorer(obs)[0]
    assert explorer.returning[0] and action[0]<0
    assert explorer._frame_end_margin(scene.uav_pos[0],.23,action)>=0
    assert explorer.counts['turnbacks']==1


def test_random_overrides_preserve_unsafe_action_distribution():
    _,scene,energy,*_=build_env()
    scene.uav_battery.fill(.001)
    obs=scene.observe_lower()
    guarded=WaypointExploration(scene,np.random.default_rng(42),1,energy_guard=True,energy_model=energy)
    legacy=WaypointExploration(scene,np.random.default_rng(42),1,energy_guard=False,energy_model=energy)
    np.testing.assert_array_equal(guarded(obs),legacy(obs))
    assert guarded.counts['random_actions']==6
    assert guarded.counts['random_infeasible_forecasts']==6
    assert not guarded.returning.any()


def test_guard_state_roundtrip_and_reset():
    _,scene,*_=build_env()
    explorer=WaypointExploration(scene,np.random.default_rng(42),energy_guard=True)
    explorer.returning[0]=True
    explorer.counts['turnbacks']=3
    state=explorer.state_dict()
    explorer.reset()
    assert not explorer.returning.any()
    explorer.load_state_dict(state)
    assert explorer.returning[0] and explorer.counts['turnbacks']==3
