"""Episode accounting and passive policy probes must preserve learning behavior."""
import numpy as np
import torch
from test_rl import trainer_config
from jv_uav.rl.trainer import JointTrainer


def test_lower_episode_return_terms_and_discount_match_observed_steps():
    env,cfg=trainer_config()
    env['time']['episode_upper_frames']=1
    trainer=JointTrainer(env,cfg)
    original=trainer.env.low_runner.on_step
    observed=[]
    def record(obs,action,result):
        observed.append((float(result.reward),dict(result.reward_terms),float(result.collected_per_sensor.sum())))
        original(obs,action,result)
    trainer.env.low_runner.on_step=record
    row=trainer.step();metrics=row['episode_metrics']
    assert metrics['lower_diagnostics_complete']
    assert np.isclose(metrics['lower_return'],sum(x[0] for x in observed))
    assert np.isclose(metrics['lower_discounted_return'],sum(cfg['sac']['gamma']**i*x[0] for i,x in enumerate(observed)))
    assert np.isclose(metrics['lower_return'],sum(metrics['lower_reward_terms_sum'].values()))
    assert np.isclose(metrics['lower_collected_packets'],sum(x[2] for x in observed))
    assert metrics['lower_recorded_steps']==len(observed)
    assert np.isclose(metrics['lower_mean_reward'],metrics['lower_return']/len(observed))


def test_policy_probes_do_not_change_rng_actions_or_optimization():
    env,cfg=trainer_config()
    results=[]
    for interval in [0,1]:
        cfg['diagnostics_every_low_steps']=interval
        trainer=JointTrainer(env,cfg)
        for _ in range(3):trainer.step()
        results.append((trainer.env.trace.to_serializable(),
                        {k:v.clone() for k,v in trainer.sac.actor.state_dict().items()},torch.get_rng_state().clone()))
    assert results[0][0]==results[1][0]
    for key in results[0][1]:torch.testing.assert_close(results[0][1][key],results[1][1][key],rtol=0,atol=0)
    assert torch.equal(results[0][2],results[1][2])


def test_legacy_partial_episode_does_not_fabricate_full_return():
    from jv_uav.rl.diagnostics import EpisodeDiagnostics
    stats=EpisodeDiagnostics(.99)
    summary=stats.summary(12)
    assert not summary['lower_diagnostics_complete']
    assert summary['lower_return'] is None
    assert summary['lower_reward_terms_sum'] is None
