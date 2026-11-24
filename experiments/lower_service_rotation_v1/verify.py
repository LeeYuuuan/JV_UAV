"""Deterministic invariants and SAC/resume checks; no long training runs."""
from pathlib import Path
import copy
import sys
import tempfile
import numpy as np
import torch
import yaml

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'src'))
from jv_uav.config import load_config
from environment import Environment, Status
from agent import SAC
from train import Trainer, validate, configure_auxiliary


def run():
    torch.set_num_threads(1)
    world=load_config(HERE/'environment.yaml')
    cfg=yaml.safe_load((HERE/'config.yaml').read_text())
    validate(cfg,world)
    stale=copy.deepcopy(cfg);stale['reward']['auxiliary_max']=5.
    fixed=configure_auxiliary(stale,.5)
    assert fixed['reward']['auxiliary_max']==.5 and stale['reward']['auxiliary_max']==5.
    assert fixed['revision']=='displacement_service_rotation_v1_aux05'
    assert configure_auxiliary(fixed,.5,resume=True)==fixed
    try:
        configure_auxiliary(stale,.5,resume=True)
    except ValueError as exc:assert 'checkpoint auxiliary_max=5.0' in str(exc)
    else:raise AssertionError('old-reward replay must not silently resume under a new ceiling')
    # Check individual physical transitions without the explicit cycle reset first.
    cfg['cycle_reset']=False
    def env():return Environment(world,cfg,123)
    e=env()
    # Actual return: 10s climb + 100s horizontal; first step cannot teleport home.
    e.status[:]=Status.IDLE;e.status[0]=Status.RETURNING
    e.xy[0]=e.home+[600,0];e.climb_left[0]=10;e.soc[0]=1
    e.step_count=1
    _,_,_,_,row=e.step(np.zeros((0,2),np.float32))
    assert np.allclose(e.xy[0],e.home+[300,0])
    assert e.status[0]==Status.RETURNING
    assert np.isclose(e.soc[0],1-(400*10+257*50)/450000,atol=1e-6)
    before=float(e.soc[0])
    e.step(np.zeros((0,2),np.float32))
    assert e.status[0]==Status.CHARGING and np.allclose(e.xy[0],e.home)
    assert np.isclose(e.soc[0],before-257*50/450000+500*10/450000,atol=1e-6)

    # FIFO slots and mid-step full-charge handoff.
    c=copy.deepcopy(cfg);c['charging_slots']=1
    e=Environment(world,c,123);e.step_count=1;e.status[:]=Status.IDLE
    e.status[0]=Status.CHARGING;e.soc[0]=1-500*10/450000
    e.status[1]=Status.WAITING;e.soc[1]=.2;e.waiting=[1]
    e.step(np.zeros((0,2),np.float32))
    assert e.status[0]==Status.IDLE and e.status[1]==Status.CHARGING
    assert np.isclose(e.soc[1],.2+500*50/450000,atol=1e-6)

    # Collection union counts each target exactly once despite six overlaps.
    e=env();e.step_count=1;e.status[:]=Status.SERVING
    e.sensor_xy[:]=e.home;e.backlog[:]=100;e.age[:]=60
    e.collector.radius=e.radius
    _,reward,terminated,truncated,row=e.step(np.zeros((6,2),np.float32))
    assert row['target_hits']==6 and row['terms']['target']==30
    assert 0<=row['terms']['auxiliary']<cfg['reward']['auxiliary_max']
    assert np.all(e.backlog==0) and np.all(e.age==0)
    assert not terminated and not truncated

    # Diagonal displacement obeys the Euclidean 180m step bound.
    e=env();e.step_count=1;ids=e.ids();before=e.xy.copy()
    e.step(np.ones((len(ids),2),np.float32))
    assert np.allclose(np.linalg.norm(e.xy[ids]-before[ids],axis=1),e.max_distance,atol=1e-3)

    # A cycle check occurs before dispatch; no implicit teleport/full refill.
    e=env();e.step_count=19;e.status[:]=Status.IDLE;e.status[0]=Status.SERVING
    e.xy[0]=e.home+[1600,0];e.soc[0]=.08
    _,_,_,_,row=e.step(np.zeros((1,2),np.float32))
    assert row['unsafe_ids']==[0] and row['terms']['return_failure']<0
    assert np.allclose(e.xy[0],e.home+[1600,0]) and e.soc[0]<.08
    # A previously charged failure is never charged again during actual return.
    e.status[:]=Status.IDLE;e.status[0]=Status.RETURNING;e.soc[0]=1e-6;e.climb_left[0]=10
    _,_,_,_,row=e.step(np.zeros((0,2),np.float32))
    assert e.status[0]==Status.STOPPED and row['penalty_ids']==[]

    # Every replay observation contains observable information only.
    e=env();o=e.observation()
    assert set(o)=={'ids','xy','soc','status','age','targets','target_age','phase'}
    assert len(o['targets'])==6 and 'backlog' not in o
    agent=SAC(e,cfg['sac'],'cpu')
    empty=copy.deepcopy(o);empty['ids']=np.array([],dtype=int);empty['status'][:]=Status.CHARGING
    batch=agent.pack([o,empty]);a,logp=agent.actor(batch)
    assert torch.isfinite(a).all() and torch.isfinite(logp).all() and torch.all(a[1]==0)
    assert logp[1]==0
    modified=a.detach().clone();modified[~batch['active']]=.95
    q1=agent.critic(batch,a.detach())[0];q2=agent.critic(batch,modified)[0]
    assert torch.allclose(q1,q2)
    transitions=[(o,np.zeros((len(o['ids']),2),np.float32),1.,empty,False),
                 (empty,np.zeros((0,2),np.float32),0.,o,False)]
    metrics=agent.update(transitions)
    assert all(np.isfinite(v) for v in metrics.values())
    metrics=agent.update([transitions[1]])
    assert all(np.isfinite(v) for v in metrics.values())

    # 300-step reset is only a truncation; all 300 next states remain valid.
    e=env()
    for _ in range(300):
        nxt,r,term,trunc,row=e.step(np.zeros((len(e.ids()),2),np.float32))
        assert not term and np.isfinite(r)
    assert trunc and e.step_count==300
    e.reset();assert np.all(e.soc==1) and np.all(e.xy==e.home)

    # The default cycle reset is part of the nonterminal physical transition.
    c=copy.deepcopy(cfg);c['cycle_reset']=True
    e=Environment(world,c,123);e.step_count=19;e.status[:]=Status.IDLE
    e.status[0]=Status.SERVING;e.xy[0]=e.home+[1600,0];e.soc[0]=.08
    e.backlog[:]=5000
    nxt,_,term,trunc,row=e.step(np.zeros((1,2),np.float32))
    assert row['unsafe_ids']==[0] and row['terms']['return_failure']<0
    assert not term and not trunc and row['cycle_reset']
    assert np.all(nxt['soc']==1) and np.all(nxt['xy']==e.home)
    assert np.max(e.backlog)>5000 and np.max(e.age)>60

    # Resume restores both physical state and exploration/replay RNGs.
    trainer=Trainer(world,cfg,'cpu')
    for _ in range(3):
        obs=trainer.env.observation();action,_,_=trainer.behavior(obs)
        nxt,r,term,_,_=trainer.env.step(action)
        trainer.replay.add(obs,action,r,nxt,term);trainer.steps+=1
    trainer.last_loss=trainer.sac.update(trainer.replay.sample(2));trainer.updates+=1
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'checkpoint.pt';trainer.save(path)
        original_obs=trainer.env.observation()
        expected_action,expected_sources,_=trainer.behavior(original_obs)
        expected=trainer.env.step(expected_action)
        restored=Trainer.load(path,'cpu')
        action,sources,_=restored.behavior(restored.env.observation())
        actual=restored.env.step(action)
        assert sources==expected_sources and np.array_equal(action,expected_action)
        assert expected[1]==actual[1]
        for key in expected[0]:assert np.array_equal(expected[0][key],actual[0][key])
    print('PASS: physical return/charging, FIFO handoff, union reward, radial limit, cycle ordering, failure deduplication, observable-only inputs, empty-service SAC, 300-step bootstrap boundary, exact CPU resume')


if __name__=='__main__':run()
