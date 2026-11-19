import torch
import numpy as np
from test_rl import trainer_config
from test_core import ROOT
from jv_uav.rl.transfer import Paper1Actor, read_paper1_actor
from jv_uav.rl.trainer import JointTrainer


def test_paper1_adapter_feature_order_and_charge_probability():
    actor = Paper1Actor(6, 16)
    soc = torch.tensor([-.1, .2, .3, .4, .5, 1.])
    obs = torch.cat([soc.expand(6, -1), torch.eye(6), torch.eye(4)[torch.arange(6)%4],
                     torch.tensor([.5, 1.]).expand(6, -1)], dim=1)
    old = actor.old_features(obs)
    for i in range(6):
        expected = torch.cat([soc[i:i+1], obs[i,12:16], obs[i,-2:], torch.tensor([i/5]), soc])
        torch.testing.assert_close(old[i], expected)
    old_prob_charge = actor.logits(actor.net(old)).softmax(-1)[:,0]
    torch.testing.assert_close(actor(obs).squeeze(-1).sigmoid(), old_prob_charge)
    loss = actor(obs).sum()
    loss.backward()
    assert all(p.grad is not None for p in actor.parameters())


def test_paper1_transfer_load_validation_and_resume():
    env, cfg = trainer_config()
    actor = Paper1Actor(6, 16)
    path = ROOT / 'tests/_paper1_transfer.pt'
    resume = ROOT / 'tests/_paper1_resume.pt'
    try:
        torch.save({'actor': actor.state_dict()}, path)
        state, metadata = read_paper1_actor(path, 6)
        try:
            read_paper1_actor(path, 7)
        except ValueError as error:
            assert 'no padding/truncation' in str(error)
        else:
            raise AssertionError('fleet mismatch accepted')
        cfg['mappo'].update(observation_mode='arrival_soc', actor_architecture='paper1_notebook', transfer_actor_hidden=16)
        cfg['upper_transfer'] = metadata
        trainer = JointTrainer(env, cfg)
        trainer.mappo.actor.load_state_dict(state)
        assert len(trainer.mappo.actor_opt.state) == 0
        trainer.step()
        trainer.save(resume)
        expected = trainer.step()
        loaded = JointTrainer.load(resume)
        assert loaded.cfg['upper_transfer'] == metadata
        assert loaded.step() == expected
    finally:
        path.unlink(missing_ok=True)
        resume.unlink(missing_ok=True)


def test_six_uav_512_actor_matches_independent_notebook_network():
    from torch import nn
    from torch.distributions import Categorical, Bernoulli
    torch.manual_seed(42)
    reference = nn.Sequential(nn.Linear(14, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                              nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 2))
    actor = Paper1Actor(6, 512)
    source = {('logits.' + k[2:]) if k.startswith('8.') else ('net.' + k): v
              for k, v in reference.state_dict().items()}
    actor.load_state_dict(source, strict=True)
    soc = torch.tensor([-.1, .2, .3, .4, .5, 1.])
    status = torch.eye(4)[torch.tensor([0, 1, 2, 0, 1, 2])]
    occupancy = torch.ones(6, 2)
    current = torch.cat([soc.expand(6, -1), torch.eye(6), status, occupancy], dim=1)
    old = torch.stack([torch.cat([soc[i:i+1], status[i], occupancy[i], torch.tensor([i/5]), soc]) for i in range(6)])
    torch.testing.assert_close(actor.old_features(current), old, rtol=0, atol=0)
    a = Categorical(logits=reference(old))
    b = Bernoulli(logits=actor(current).squeeze(-1))
    torch.testing.assert_close(a.probs[:, 0], b.probs)
    torch.testing.assert_close(a.entropy(), b.entropy())
    actions = torch.tensor([0, 1, 0, 1, 1, 0])
    torch.testing.assert_close(a.log_prob(1-actions), b.log_prob(actions.float()))
