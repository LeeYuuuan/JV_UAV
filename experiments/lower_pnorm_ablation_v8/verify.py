"""Numerical reward, physical ablation, observation and exact-resume checks."""
from pathlib import Path
import copy,sys,tempfile
import numpy as np
import torch,yaml
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[1]/'src')]
from jv_uav.config import load_config
from environment import Environment,Status
from agent import SAC
from train import Trainer,configure_reward,validate

def run():
    torch.set_num_threads(1)
    world=load_config(HERE/'environment.yaml');raw=yaml.safe_load((HERE/'config.yaml').read_text())
    def cfg(mode):return configure_reward(raw,energy_mode=mode)
    def env(mode='real'):
        c=cfg(mode);validate(c,world);return Environment(world,c,123)
    def no_arrivals(e):
        e.world['sensors']['arrival_rate_per_sec']=0.;e.status[:]=Status.IDLE;e.age[:]=0
    for mode in ['real','none']:
        # Exact generalized p-mean: all equal vs one isolated large queue.
        e=env(mode);no_arrivals(e);e.backlog[:]=1000.
        r=e.step(np.zeros((0,2),np.float32))[4]
        assert np.isclose(r['terms']['task'],-.5) and set(r['terms'])=={'task','boundary','return_failure'}
        e=env(mode);no_arrivals(e);e.backlog[:]=0;e.backlog[0]=2000.
        r=e.step(np.zeros((0,2),np.float32))[4]
        assert np.isclose(r['task_cost'],(1/e.m)**(1/8))
        # Real full clear is zero task cost, without hidden collection/round bonus.
        e=env(mode);no_arrivals(e);e.sensor_xy[:]=e.home;e.backlog[:]=2000.
        e.status[0]=Status.SERVING
        r=e.step(np.zeros((1,2),np.float32))[4]
        assert r['round_completed'] and r['reward']==0 and not e.backlog.any()
        e.rows.clear();assert e.step(np.zeros((1,2),np.float32))[1]==0
        # 300 is only a reporting boundary; sensor state continues.
        e=env(mode);e.step_count=299;e.sensor_xy[:]=e.home+[1500,1500];e.backlog[:]=5000;e.age[:]=2000
        n,r,end,report,row=e.step(np.zeros((len(e.ids()),2),np.float32))
        assert report and not end and e.backlog.min()>=5000 and e.age.min()>2000
        e.rows.clear();e.step(np.zeros((len(e.ids()),2),np.float32));assert e.step_count==301
        # No phase input; all three age paths use the same bounded map.
        e=env(mode);o=e.observation();a=SAC(e,e.cfg['sac'],'cpu');p=a.pack([o])
        assert 'phase' not in o and 'phase' not in p and not hasattr(a.actor.encoder,'phase')
        assert p['uav'].shape==(1,6,14) and p['age'].shape==(1,301)
        o['age'][:]=6000.;o['assigned_age'][:]=6000.;o['target_age'][:]=6000.
        p=a.pack([o]);assert torch.allclose(p['age'][0,:150],torch.full((150,),.5))
        assert (p['uav'][0,:,12]==.5).all() and (p['targets'][0,:,2]==.5).all()
        assert np.allclose(a.normalize_age([0,6000,6e6]),[0,.5,1000/1001])
    # Both modes start from exactly matching actor and critic parameters.
    t1=Trainer(world,cfg('real'),'cpu');t2=Trainer(world,cfg('none'),'cpu')
    for k,v in t1.sac.actor.state_dict().items():assert torch.equal(v,t2.sac.actor.state_dict()[k])
    for k,v in t1.sac.critic.state_dict().items():assert torch.equal(v,t2.sac.critic.state_dict()[k])
    # Identical bounded180m motion, only real mode spends SOC.
    ends=[]
    for mode in ['real','none']:
        e=env(mode);e.status[:]=Status.IDLE;e.status[0]=Status.SERVING
        row=e.step(np.ones((1,2),np.float32))[4];ends.append(row['position_after'][0])
        assert np.isclose(np.linalg.norm(e.xy[0]-e.home),180,atol=.001)
        assert (e.soc[0]<1) if mode=='real' else np.all(e.soc==1)
    assert np.allclose(*ends)
    # Only real energy penalizes unsafe terminal positions.
    for mode in ['real','none']:
        for inward in [False,True]:
            e=env(mode);e.step_count=19;e.status[:]=Status.IDLE;e.status[0]=Status.SERVING
            e.xy[0]=e.home+[500,0];e.soc[0]=.1 if mode=='real' else 1.
            row=e.step(np.array([[-1 if inward else 1,0]],np.float32))[4]
            assert row['terms']['return_failure']==(-5. if mode=='real' and not inward else 0.)
    # No-energy avoids energy-update methods and still returns over real time.
    e=env('none');e.cfg['service_count_min']=e.cfg['service_count_max']=6
    def forbidden(*args,**kwargs):raise AssertionError('energy evolution in no-energy mode')
    e.energy.return_trip=forbidden;e.energy.serving_energy_fraction=forbidden
    for _ in range(61):
        row=e.step(np.ones((len(e.ids()),2),np.float32))[4];assert np.all(e.soc==1)
        assert not row['penalty_ids'] and not row['depleted_ids'] and not row['return_failed_ids']
    e=env('none');e.status[:]=Status.IDLE;e.status[0]=Status.RETURNING
    e.xy[0]=e.home+[600,0];e.climb_left[0]=10
    e.step(np.zeros((0,2),np.float32));assert np.allclose(e.xy[0],e.home+[300,0]) and e.soc[0]==1
    e.step(np.zeros((0,2),np.float32));assert np.array_equal(e.xy[0],e.home) and e.status[0]==Status.IDLE
    # SAC updates and exact resume with RNG/replay restoration in both modes.
    for mode in ['real','none']:
        t=Trainer(world,cfg(mode),'cpu')
        for _ in range(3):
            o=t.env.observation();a,_,_=t.behavior(o);n,r,end,_,_=t.env.step(a)
            t.replay.add(o,a,r,n,end);t.steps+=1
        assert all(np.isfinite(x) for x in t.sac.update(t.replay.sample(2)).values())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pt';t.save(path)
            a,sources,_=t.behavior(t.env.observation());expected=t.env.step(a)
            restored=Trainer.load(path,'cpu');b,sources2,_=restored.behavior(restored.env.observation());actual=restored.env.step(b)
            assert np.array_equal(a,b) and sources==sources2 and actual[1]==expected[1]
            for k in actual[0]:assert np.array_equal(expected[0][k],actual[0][k])
        try:configure_reward(cfg(mode),resume=True,energy_mode='none' if mode=='real' else 'real')
        except ValueError:pass
        else:raise AssertionError('energy mode switched on resume')
    print('PASS: exact p-mean, no hidden bonuses, bounded/no-phase input, real/none physical isolation, matching initial weights, SAC and exact resume')

if __name__=='__main__':run()
