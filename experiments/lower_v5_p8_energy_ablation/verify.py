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
from train import Trainer, validate, configure_reward


def run():
    torch.set_num_threads(1)
    world=load_config(HERE/'environment.yaml')
    cfg=yaml.safe_load((HERE/'config.yaml').read_text())
    # Exercise archived max-mode invariants below; P8 differential tests are separate.
    cfg['reward']['cost_mode']='max'
    validate(cfg,world)
    base_cfg=copy.deepcopy(cfg);base_cfg['reward']['cost_mode']='max'
    fixed=configure_reward(base_cfg,cost_mode='pnorm')
    assert fixed['reward']['cost_mode']=='pnorm' and base_cfg['reward']['cost_mode']=='max'
    assert configure_reward(fixed,resume=True)==fixed
    try:
        configure_reward(fixed,resume=True,cost_mode='max')
    except ValueError as exc:assert 'resume reward mismatch' in str(exc)
    else:raise AssertionError('reward changes must not reuse old replay')
    for key,value in [('potential_beta',9.),('cost_p',4.),('backlog_reference_packets',1000.)]:
        try:configure_reward(cfg,resume=True,**{key:value})
        except ValueError:pass
        else:raise AssertionError('resume reward mismatch accepted')
    # Check individual physical transitions without the explicit cycle reset first.
    cfg['cycle_reset']=False
    def env():return Environment(world,cfg,123)
    # Reward follows the realized move/collection, not merely the pre-action state.
    outcomes=[]
    for action in (np.zeros((1,2),np.float32),np.array([[1,0]],np.float32)):
        e=env();e.step_count=1;e.status[:]=Status.IDLE;e.status[0]=Status.SERVING
        e.sensor_xy[:]=e.home+[1500,1500];e.sensor_xy[0]=e.home+[180,0]
        row=e.step(action)[4];outcomes.append(row)
    assert outcomes[0]['target_hits']==0
    assert outcomes[1]['target_hits']==1 and outcomes[1]['backlog'][0]==0
    assert 'target' not in outcomes[1]['terms'] and 'auxiliary' not in outcomes[1]['terms']
    for row in outcomes:
        assert np.isclose(row['terms']['task'],-max(row['backlog'])/cfg['reward']['backlog_reference_packets'])
        assert np.isclose(row['terms']['shaping'],cfg['reward']['potential_beta']*(cfg['sac']['gamma']*row['potential_after']-row['potential_before']))
        assert np.isclose(row['reward'],sum(row['terms'].values()))
    # Linear scale is configuration-driven; safe states cost zero.
    e=env();e.status[:]=Status.IDLE;e.status[0]=Status.SERVING
    e.xy[0]=e.home+[1000,0]
    required=float(e.return_requirement(e.xy[0])[0])
    e.soc[0]=required+.01;assert e.linear_energy_cost()[2][0]==0
    coefficient=cfg['reward']['energy_deficit_weight']/cfg['reward']['energy_deficit_scale_soc']
    e.soc[0]=required-.02;assert np.isclose(e.linear_energy_cost()[2][0],.02*coefficient,atol=1e-5)
    e.soc[0]=required-.04;assert np.isclose(e.linear_energy_cost()[2][0],.04*coefficient,atol=1e-5)
    e.xy[0]=e.home+[1800,0]
    assert np.isclose(e.failure_cost([0]),.75*(10+5))
    # Equal-energy inward/outward actions are distinguished immediately.
    costs=[]
    for sign in (-1,1):
        e=env();e.step_count=17;e.status[:]=Status.IDLE;e.status[0]=Status.SERVING
        unit=np.ones(2,dtype=np.float32)/np.sqrt(2.)
        e.xy[0]=e.home+unit*2000;e.soc[0]=.2
        row=e.step((sign*unit).reshape(1,2))[4]
        costs.append(row['energy_cost_by_uav'][0])
    assert np.isclose(costs[0],.0209926*coefficient,atol=1e-3)
    assert np.isclose(costs[1],.0552593*coefficient,atol=1e-3) and costs[0]<costs[1]
    # Exhaustion does not turn off mission energy cost; cycle reset does.
    e=env();e.step_count=1;e.status[:]=Status.IDLE;e.status[0]=Status.SERVING
    e.xy[0]=e.home+[1000,0];e.soc[0]=1e-6
    row=e.step(np.zeros((1,2),np.float32))[4]
    assert e.status[0]==Status.STOPPED and row['terms']['energy_deficit']<0
    row2=e.step(np.zeros((0,2),np.float32))[4]
    assert np.isclose(row2['terms']['energy_deficit'],row['terms']['energy_deficit'])
    # Remaining-return requirement includes unfinished climb only.
    e.status[0]=Status.RETURNING;e.return_started[0]=True;e.climb_left[0]=5
    expected=(257*1000/6+400*5)/450000
    assert np.isclose(e.linear_energy_cost()[0][0],expected)
    # Pure, repeatable assignment: age outranks distance, ties use distance.
    e=env();e.status[:]=Status.IDLE;e.status[[0,1]]=Status.SERVING
    e.xy[0]=e.home;e.xy[1]=e.home+[1000,0]
    e.age[:]=100;e.age[:3]=300
    e.sensor_xy[0]=e.home+[100,0];e.sensor_xy[1]=e.home+[900,0]
    e.sensor_xy[2]=e.home+[0,1500]
    assigned,order=e.assignment()
    assert order.tolist()==[0,1] and assigned[:2].tolist()==[0,1]
    assert np.array_equal(assigned,e.assignment()[0])
    e.age[2]=400
    assert e.assignment()[0][0]==2
    e.step_count=10;assert e.assignment()[1].tolist()==[1,0]
    # Targets rebind when positions change; there is no hidden observation RNG.
    e.step_count=0;e.age[:3]=300;e.xy[0]=e.home+[850,0]
    assert e.assignment()[0][0]==1
    e.age[:]=100;e.age[:2]=300;e.age[2:9]=200
    eligible,threshold=e.reward_targets()
    assert threshold==200 and eligible.tolist()==list(range(9))
    obs=e.observation()
    assert obs['target_mask'].sum()==2 and obs['assigned_ids'][2:].tolist()==[-1]*4
    e=env()
    # Actual return: 10s climb + 100s horizontal; first step cannot teleport home.
    e.status[:]=Status.IDLE;e.status[0]=Status.RETURNING
    e.xy[0]=e.home+[600,0];e.climb_left[0]=10;e.soc[0]=1
    e.step_count=1
    _,_,_,_,row=e.step(np.zeros((0,2),np.float32))
    assert np.allclose(e.xy[0],e.home+[300,0])
    assert e.status[0]==Status.RETURNING
    assert np.isclose(e.soc[0],1-(400*10+257*50)/450000,atol=1e-6)
    assert np.isclose(row['required_return_soc'][0],257*50/450000,atol=1e-6)
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
    assert row['target_hits']==e.m and row['terms']['task']==0
    assert np.all(e.backlog==0) and np.all(e.age==0)
    assert not terminated and not truncated
    # Reward uses the frozen pre-action K-th-age tie set, not selected coordinates.
    e=env();e.step_count=1;e.status[:]=Status.SERVING;e.sensor_xy[:]=e.home
    e.age[:]=100;e.age[:2]=300;e.age[2:9]=200
    _,_,_,_,row=e.step(np.zeros((6,2),np.float32))
    assert row['target_age_threshold_sec']==200 and row['target_hits']==9
    assert row['eligible_target_count']==9 and row['terms']['task']==0

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
    assert set(o)=={'ids','xy','soc','status','age','targets','target_age','phase','target_mask',
                   'target_ids','assigned_ids','assigned_xy','assigned_age','assigned_valid','assignment_order'}
    assert len(o['targets'])==6 and 'backlog' not in o
    agent=SAC(e,cfg['sac'],'cpu')
    e.status[:]=Status.CHARGING;empty=e.observation()
    assert not empty['target_mask'].any() and not empty['assigned_valid'].any()
    batch=agent.pack([o,empty]);a,logp=agent.actor(batch)
    assert torch.isfinite(a).all() and torch.isfinite(logp).all() and torch.all(a[1]==0)
    assert logp[1]==0
    packed=agent.pack([o]);ids=o['ids']
    assert packed['uav'].shape[-1]==14
    assert np.allclose(packed['uav'][0,ids,10:12].numpy(),(o['assigned_xy'][ids]-o['xy'][ids])/e.size)
    # Padded sensor coordinates have no policy effect, including empty-service states.
    padded=agent.pack([o,empty]);changed={k:v.clone() for k,v in padded.items()}
    changed['targets'][~changed['target_mask']]=100000
    assert torch.allclose(agent.actor(padded,True)[0],agent.actor(changed,True)[0])
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
    assert row['terms']['energy_deficit']<0 and np.all(e.linear_energy_cost()[2]==0)
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
    # Max and smooth p=8 cost: finite, all-sensor cost, no clipping or positive bonuses.
    e=env();e.cfg['reward']['cost_mode']='max';values=np.array([1000.,2000.,0.])
    assert e.backlog_cost(values)==1.
    e.cfg['reward']['cost_mode']='pnorm'
    assert np.isclose(e.backlog_cost(values),np.mean((values/2000.)**8)**(1/8))
    assert e.backlog_cost(np.array([0.,0.]))==0
    assert np.isfinite(e.backlog_cost(np.array([1e100,1e100])))
    assert np.isclose(e.backlog_cost(np.full(150,2000.)),1.)
    assert e.backlog_cost(np.full(150,10000.))>1
    # Unchanged goal, equal energy: inward has a larger shaping reward than hold/outward.
    shaped=[]
    for dx in (1.,0.,-1.):
        e=env();e.step_count=1;e.status[:]=Status.IDLE;e.status[0]=Status.SERVING
        e.age[:]=0;e.age[0]=1000;e.sensor_xy[0]=e.home+[1200,0]
        phi=e.potential();row=e.step(np.array([[dx,0]],np.float32))[4]
        assert 0<=phi<=1 and 0<=row['potential_after']<=1
        shaped.append(row['terms']['shaping'])
    assert shaped[0]>shaped[1]>shaped[2]
    e.status[:]=Status.CHARGING;assert e.potential()==0
    # All300 transitions, including rebindings, dispatch/home resets and cutoff,
    # must telescope using the exact replay successor. No special terminal zero.
    c=copy.deepcopy(cfg);c['cycle_reset']=True;e=Environment(world,c,123)
    initial_phi=e.potential();gamma=c['sac']['gamma'];beta=c['reward']['potential_beta'];discounted=0.
    for j in range(300):
        ob=e.observation();act=np.zeros((len(ob['ids']),2),np.float32)
        nxt,_,term,trunc,row=e.step(act)
        assert np.isclose(row['potential_after'],e.potential(nxt))
        discounted+=gamma**j*row['terms']['shaping']
        if row['cycle_reset']:
            assert np.all(nxt['xy']==e.home) and np.all(nxt['soc']==1)
            assert row['shaping_at_dispatch']
        if j==299:assert trunc and not term and row['potential_after']>0
    assert np.isclose(discounted,beta*(-initial_phi+gamma**300*e.potential()),atol=1e-8)
    # Pure shaping cannot improve an exact bootstrapped task objective:
    # V_shaped(s)=V_task(s)-beta*Phi(s), including cutoff residual.
    arbitrary_terminal_value=12.
    assert np.isclose(discounted+gamma**300*(arbitrary_terminal_value-beta*e.potential()),
                      gamma**300*arbitrary_terminal_value-beta*initial_phi,atol=1e-8)
    print('PASS: post-collection max/p8 costs, no old bonuses, direction shaping, target changes/reset/cutoff telescoping, safety physics, masks/SAC and exact resume')


if __name__=='__main__':run()
