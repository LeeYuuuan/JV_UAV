import numpy as np
from test_rl import trainer_config, ROOT
from jv_uav.rl.trainer import JointTrainer
from jv_uav.rl.observations import upper_observation


def test_legacy_checkpoint_keeps_original_features_and_resumes():
    env, cfg = trainer_config()
    cfg['mappo'].pop('observation_mode')
    trainer = JointTrainer(env, cfg)
    obs = upper_observation(trainer.env.scene, trainer.env.energy_model, trainer.mappo.observation_mode)
    assert obs['actor'].shape == (6, 19) and obs['critic'].shape == (38,)
    np.testing.assert_array_equal(obs['actor'][0, :6], trainer.env.scene.uav_battery)
    trainer.step()
    path = ROOT / 'tests/_legacy_upper_resume.pt'
    try:
        trainer.save(path)
        expected = trainer.step()
        resumed = JointTrainer.load(path)
        assert resumed.mappo.observation_mode == 'current_soc_return_time'
        assert resumed.step() == expected
    finally:
        path.unlink(missing_ok=True)
