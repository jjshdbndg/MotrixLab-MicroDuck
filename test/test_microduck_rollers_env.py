import numpy as np

from motrix_envs import registry
from motrix_envs.locomotion.microduck.rollers_cfg import MicroduckRollersNpEnvCfg

ENV_NAME = "microduck-flat-tile-passive-rollers"


def test_roller_environment_contract_and_step():
    env = registry.make(ENV_NAME, sim_backend="np", num_envs=2)
    assert env.observation_space.shape == (61,)
    assert env.action_space.shape == (14,)
    state = env.step(np.zeros((2, 14), dtype=np.float32))
    assert state.obs.shape == (2, 61)
    assert state.reward.shape == (2,)
    assert np.all(np.isfinite(state.obs))
    assert np.all(np.isfinite(state.reward))
    assert env.model.num_actuators == 14
    assert not any(name.startswith("passive_") for name in env.model.actuator_names)
    assert "left_push_fraction" in state.info["metrics"]
    assert "right_push_fraction" in state.info["metrics"]
    assert "push_switch_rate" in state.info["metrics"]
    assert "right_lift_fraction" in state.info["metrics"]
    assert "right_complete_fraction" in state.info["metrics"]


def test_roller_config_prioritizes_right_support_release_and_heading():
    cfg = MicroduckRollersNpEnvCfg()
    assert cfg.reward.support_switch_weight >= 0.75
    assert cfg.reward.scales["track_angular_velocity"] >= 1.0
    assert cfg.reward.stroke_cycle_weight > 1.0
