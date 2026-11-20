from pathlib import Path
import copy,tempfile,json,sys
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments/lower_waypoint_persistent')]
from jv_uav.config import load_config
from persistent_env import PersistentEnv
from continuing_trainer import ContinuingTrainer
from waypoint_sampler import WaypointSampler
from train import validate


def setup(tiny=False):
    ec=load_config(ROOT/'configs/env_lower_continuing6.yaml')
    cfg=yaml.safe_load((ROOT/'configs/training_lower_continuing6_greedy_rho08.yaml').read_text())
    if tiny:cfg['sac'].update(hidden=16,heads=2,batch_size=2,replay_capacity=128)
    validate(cfg);return ec,cfg


def test_continuing_environment_crosses_15_and_30_without_reset():
    ec,cfg=setup();ec['sensors']['arrival_rate_per_sec']=0
    env=PersistentEnv(ec,cfg);env.scene.sensor_backlog[:]=np.arange(6)+100
    initial=env.scene.sensor_backlog.copy()
    for k in range(31):
        _,_,done,row=env.step(np.zeros((1,2),np.float32))
        assert not done and not row['done']
        assert env.steps==k+1 and env.scene.now_sec==(k+1)*60
        assert np.array_equal(initial,env.scene.sensor_backlog)
        assert np.all(env.scene.sensor_last_visit_sec==60+(k+1)*60)


def test_windows_preserve_scene_and_all_replay_bootstraps():
    ec,cfg=setup(True);a=ContinuingTrainer(ec,cfg)
    for _ in range(15):row=a.step()
    assert row['statistics_window_end'] and a.windows==1 and a.window_pending
    state=copy.deepcopy(a.env.state_dict());r=a.step()
    # Replay source at step16 must be the final observation at step15, not reset.
    obs,_,_,nxt,done,_=a.replay.items[-1]
    assert np.array_equal(obs['all_positions'],state['scene']['uav_pos'])
    assert np.array_equal(obs['sensor_last_visit_sec'],state['scene']['sensor_last_visit_sec'])
    assert a.env.steps==16 and len(a.env.records)==1 and not r['statistics_window_end']
    assert all(not x[4] for x in a.replay.items)
    boundary=a.replay.items[14]
    result=a.sac.update([boundary,boundary])
    assert np.isclose(result['discount_mean'],.95)


def test_continuing_exact_resume_at_window_and_inside_window():
    ec,cfg=setup(True)
    for count in [15,17]:
        a=ContinuingTrainer(ec,cfg)
        for _ in range(count):a.step()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'snapshot.pt';a.save(path)
            ra=a.step();b=ContinuingTrainer.load(path);rb=b.step()
            assert ra['action']==rb['action'] and ra['reward']==rb['reward']
            assert np.array_equal(a.env.scene.sensor_backlog,b.env.scene.sensor_backlog)
            assert a.windows==b.windows and a.window_pending==b.window_pending
            assert np.isclose(ra['losses']['critic_loss'],rb['losses']['critic_loss'])


def test_top18_original_reward_rho08_and_topk_exclusion():
    ec,cfg=setup();ec['sensors']['arrival_rate_per_sec']=0
    cfg['reward']['movement_weight']=0
    env=PersistentEnv(ec,cfg);env.scene.sensor_backlog[:]=[100,200,300,400,500,600]
    target=(env.scene.sensor_pos[:1]/300-1).astype(np.float32)
    _,r,_,_=env.step(target)
    expected=-600/300*(1-.8*100/2100)
    assert env.records[-1]['covered_ids']==[0] and np.isclose(r,expected)
    assert np.isclose(env.records[-1]['reward_clear_fraction'],100/2100)
    cfg['reward']['top_k']=2
    env=PersistentEnv(ec,cfg);env.scene.sensor_backlog[:]=[100,200,300,400,500,600]
    _,r,_,_=env.step(target)
    assert r==-2 and env.records[-1]['reward_clear_fraction']==0
    assert env.records[-1]['reward_topk_sensor_ids']==[5,4]


def test_eval_two_pages_are_one_continuous_rollout_and_do_not_touch_training():
    ec,cfg=setup(True);t=ContinuingTrainer(ec,cfg)
    t.step();before=t.env.state_dict();torch_before=torch.get_rng_state().clone()
    with tempfile.TemporaryDirectory() as folder:
        summary=t.evaluate(1001,folder,mode='greedy_reference')
        assert summary['steps']==30 and not summary['terminated']
        assert len(summary['segment_summaries'])==2
        assert (Path(folder)/'eval_steps_01_15.png').exists()
        assert (Path(folder)/'eval_steps_16_30.png').exists()
        rows=json.loads((Path(folder)/'trace.json').read_text())['rows']
        assert [r['step'] for r in rows]==list(range(1,31))
        assert rows[14]['positions_end']==rows[15]['positions_start']
        assert not any(r['done'] for r in rows)
        assert summary['sensor_coverage_fraction']==1
    assert t.env.steps==1 and np.array_equal(before['scene']['sensor_backlog'],t.env.scene.sensor_backlog)
    assert torch.equal(torch_before,torch.get_rng_state())


def test_finite_checkpoint_not_silently_imported_into_continuing():
    with tempfile.TemporaryDirectory() as folder:
        p=Path(folder)/'old.pt';torch.save({'format':'persistent_waypoint_lastvisit_active_v3'},p)
        try:ContinuingTrainer.load(p)
        except ValueError as err:assert 'finite-horizon' in str(err)
        else:raise AssertionError('finite checkpoint accepted')


def test_large_overnight_macro_discount_groups_and_eval_pages():
    ec=load_config(ROOT/'configs/default.yaml')
    cfg=yaml.safe_load((ROOT/'configs/training_continuing_large_overnight.yaml').read_text())
    validate(cfg)
    assert cfg['total_steps']==900000 and cfg['exploration']['candidate_count']==6
    assert cfg['reward']['top_k']==18 and cfg['reward']['rho']==.8
    tiny=copy.deepcopy(cfg);tiny['sac'].update(hidden=16,heads=2,batch_size=2,replay_capacity=128)
    t=ContinuingTrainer(ec,tiny)
    env=PersistentEnv(ec,cfg);obs=env.observation();action=np.ones((6,2),np.float32)
    nxt,reward,done,row=env.step(action)
    assert row['physical_steps']==10 and not done and len(nxt['ids']) in (3,4)
    loss=t.sac.update([(obs,action,reward,nxt,False,row['duration_sec'])]*2)
    assert np.isclose(loss['discount_mean'],.95**10)
    with tempfile.TemporaryDirectory() as folder:
        summary=t.evaluate(1001,folder,mode='greedy_reference')
        rows=json.loads((Path(folder)/'trace.json').read_text())['rows']
        assert len(rows)==40 and rows[19]['positions_end']==rows[20]['positions_start']
        assert (Path(folder)/'eval_steps_01_20.png').exists()
        assert (Path(folder)/'eval_steps_21_40.png').exists()
        assert [v['steps'] for v in summary['segment_summaries']]==[20,20]


if __name__=='__main__':
    torch.set_num_threads(1)
    names=[]
    for name,fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):fn();names.append(name);print('PASS',name,flush=True)
    print(json.dumps(dict(passed=len(names),tests=names)))
