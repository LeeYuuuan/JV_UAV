"""Renewal-round anti-farming, continuous-boundary, physical and SAC checks."""
from pathlib import Path
import copy,json,sys,tempfile
import numpy as np
import torch,yaml
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[1]/'src')]
from jv_uav.config import load_config
from environment import Environment,Status
from agent import SAC
from train import Trainer,validate,configure_reward

def run():
    torch.set_num_threads(1);world=load_config(HERE/'environment.yaml')
    cfg=yaml.safe_load((HERE/'config.yaml').read_text());validate(cfg,world)
    def env():return Environment(world,cfg,123)
    def isolated():
        e=env();e.status[:]=Status.IDLE;e.status[0]=Status.SERVING;e.step_count=1
        e.sensor_xy[:]=e.home+[1500,1500];e.sensor_xy[0]=e.home;e.age[:]=60
        return e
    # One easy sensor can only increase this round's progress ONCE.
    e=isolated();phi0=e.potential()
    _,_,_,_,r=e.step(np.zeros((1,2),np.float32))
    assert r['round_new_sensor_ids']==[0] and r['terms']['round']==0
    assert np.isclose(e.round_visited.mean(),1/e.m)
    for _ in range(6):
        _,_,_,_,r=e.step(np.zeros((1,2),np.float32))
        assert r['round_new_sensor_ids']==[] and r['terms']['round']==0
    # Completed sensors excluded even when older; duplicate goals only at round tail.
    e=isolated();e.round_visited[:]=True;e.round_visited[[1,2]]=False
    e.status[:]=Status.SERVING;e.age[:]=100000;e.age[[1,2]]=1
    assigned,_=e.assignment();assert set(assigned)=={1,2} and np.all(assigned>=0)
    e.round_visited[:]=False;assigned,_=e.assignment();assert len(set(assigned))==e.n
    # Overlap clears union once; finishing only pays ONE round, no next-round overflow.
    e=env();e.sensor_xy[:]=e.home;e.status[:]=Status.SERVING
    nxt,r,term,trunc,row=e.step(np.zeros((6,2),np.float32))
    assert len(row['round_new_sensor_ids'])==e.m and row['completed_rounds']==1
    assert row['terms']['round']==5 and not e.round_visited.any()
    assert row['terms']['round']+row['terms']['shaping']>=-1e-6
    assert row['round_progress_after_collection']==1 and row['round_progress_next']==0
    # A single final missing sensor has the same completion guarantee.
    e=isolated();e.round_visited[:]=True;e.round_visited[0]=False
    nxt,_,_,_,r=e.step(np.zeros((1,2),np.float32))
    assert r['round_completed'] and r['terms']['round']==5 and not e.round_visited.any()
    assert r['terms']['round']+r['terms']['shaping']>=-1e-6
    _,_,_,_,r2=e.step(np.zeros((1,2),np.float32))
    assert r2['round_new_sensor_ids']==[0] and not r2['round_completed']
    # Real20-step UAV reset preserves mask/backlog/age;300 is reporting ONLY.
    for boundary in [20,300,600]:
        e=env();e.step_count=boundary-1;e.round_visited[0]=True
        e.sensor_xy[:]=e.home+[1500,1500];e.backlog[:]=5000;e.age[:]=2000
        before=e.potential();nxt,_,term,trunc,row=e.step(np.zeros((len(e.ids()),2),np.float32))
        assert e.round_visited[0] and e.backlog.min()>=5000 and e.age.min()>2000
        assert np.all(nxt['xy']==e.home) and np.all(nxt['soc']==1)
        assert not term and trunc==(boundary%300==0)
        assert np.isclose(row['potential_after'],e.potential(nxt))
        # The next step is legal; no external reset of the task state.
        e.rows.clear();e.step(np.zeros((len(e.ids()),2),np.float32));assert e.step_count==boundary+1
    # With identical targets, inward progress > standing > retreat.
    values=[]
    for dx in [1.,0.,-1.]:
        e=isolated();e.sensor_xy[0]=e.home+[1200,0];e.age[0]=1000
        row=e.step(np.array([[dx,0]],np.float32))[4];values.append(row['terms']['shaping'])
    assert values[0]>values[1]>values[2]
    # Encoder explicitly sees both mask and progress, not actual backlog.
    e=env();agent=SAC(e,cfg['sac'],'cpu');o=e.observation();packed=agent.pack([o])
    assert packed['age'].shape==(1,2*e.m+1) and 'backlog' not in o
    assert np.array_equal(packed['age'][0,e.m:2*e.m].numpy(),o['round_visited'])
    e.status[:]=Status.CHARGING;empty=e.observation()
    actions,lp=agent.actor(agent.pack([o,empty]));assert torch.isfinite(actions).all() and torch.all(actions[1]==0) and lp[1]==0
    transitions=[(o,np.zeros((len(o['ids']),2),np.float32),1.,empty,False),
                 (empty,np.zeros((0,2),np.float32),0.,o,False)]
    assert all(np.isfinite(v) for v in agent.update(transitions).values())
    # Safety is one-time, no repeated deficit cost, real return consumes energy.
    e=isolated();e.xy[0]=e.home+[1000,0];e.soc[0]=1e-6
    row=e.step(np.zeros((1,2),np.float32))[4]
    assert row['terms']['energy_deficit']==0 and e.status[0]==Status.STOPPED
    assert e.failure_cost([0])==5
    e=isolated();e.status[0]=Status.RETURNING;e.xy[0]=e.home+[600,0];e.climb_left[0]=10
    e.step(np.zeros((0,2),np.float32));assert np.allclose(e.xy[0],e.home+[300,0])
    # Full600-step greedy run includes rounds, target rebindings and300 boundaries.
    e=env();initial=e.potential();discounted_shape=0.;base=0.;full=0.
    for j in range(600):
        o=e.observation();delta=o['assigned_xy'][o['ids']]-o['xy'][o['ids']]
        action=delta/np.maximum(e.max_distance,np.linalg.norm(delta,axis=1))[:,None]
        nxt,r,term,trunc,row=e.step(action);assert not term
        assert np.isclose(e.potential(nxt),row['potential_after'])
        discounted_shape+=cfg['sac']['gamma']**j*row['terms']['shaping']
        base+=cfg['sac']['gamma']**j*(r-row['terms']['shaping']);full+=cfg['sac']['gamma']**j*r
    expected=-initial+cfg['sac']['gamma']**600*e.potential()
    assert abs(discounted_shape-expected)<1e-8 and e.completed_rounds>=1
    assert np.isclose(full-cfg['sac']['gamma']**600*e.potential(),base-initial)
    assert abs(e.summary()['shaping_telescoping_error'])<1e-8
    # Incomplete easy-only collection cannot earn round rewards or infinite progress.
    e=isolated();initial=e.potential();shaping_sum=0.
    for t in range(6):
        row=e.step(np.zeros((1,2),np.float32))[4]
        shaping_sum+=cfg['sac']['gamma']**t*row['terms']['shaping']
    assert e.completed_rounds==0 and e.round_visited.sum()==1
    assert abs(shaping_sum-(-initial+cfg['sac']['gamma']**6*e.potential()))<1e-8
    # Resume restores pending list, round clock, optimizer and RNG exactly.
    trainer=Trainer(world,cfg,'cpu');trainer.env.round_visited[0]=True
    for _ in range(3):
        o=trainer.env.observation();a,_,_=trainer.behavior(o);n,r,term,_,_=trainer.env.step(a)
        trainer.replay.add(o,a,r,n,term);trainer.steps+=1
    trainer.last_loss=trainer.sac.update(trainer.replay.sample(2));trainer.updates+=1
    with tempfile.TemporaryDirectory() as tmp:
        file=Path(tmp)/'checkpoint.pt';trainer.save(file)
        a,sources,_=trainer.behavior(trainer.env.observation());expected=trainer.env.step(a)
        restored=Trainer.load(file,'cpu');b,sources2,_=restored.behavior(restored.env.observation());actual=restored.env.step(b)
        assert sources==sources2 and np.array_equal(a,b) and expected[1]==actual[1]
        for k in expected[0]:assert np.array_equal(expected[0][k],actual[0][k])
    try:configure_reward(cfg,resume=True,round_reward=10.)
    except ValueError:pass
    else:raise AssertionError('changed reward accepted on resume')
    print('PASS: round union/anti-repeat/completion, pending goals,20/300 continuity, telescoping, mask inputs, physical safety, SAC and exact resume')

if __name__=='__main__':run()
