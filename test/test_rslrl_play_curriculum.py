from motrix_envs import registry
from motrix_rl.rslrl.torch.train.ppo import Trainer


def test_playback_prepares_curriculum_environment_for_final_commands():
    """Catch playback silently starting a late-stage policy at stage zero."""
    env = registry.make(
        "microduck-flat-tile-passive-rollers", sim_backend="np", num_envs=1
    )
    trainer = Trainer(
        "microduck-flat-tile-passive-rollers", sim_backend="np", enable_render=False
    )

    prepare = getattr(trainer, "_prepare_play_env", lambda candidate: None)
    prepare(env)

    assert env._command_stage() == 2
