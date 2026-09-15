"""Passive-roller MicroDuck task on smooth tile."""

from dataclasses import dataclass, field
from pathlib import Path

from motrix_envs import registry
from motrix_envs.locomotion.microduck.cfg import (
    AssetCfg,
    CommandCfg,
    ControlCfg,
    InitStateCfg,
    MicroduckWalkNpEnvCfg,
    RewardCfg,
)

ENV_NAME = "microduck-flat-tile-passive-rollers"
MODEL_FILE = str(Path(__file__).parent / "xmls" / "scene_rollers.xml")


@dataclass
class RollerCommandCfg(CommandCfg):
    # The previous 0.45–0.55 m/s target was beyond the stable passive-roller
    # regime and encouraged pitching/foot flutter. Keep the final curriculum
    # stage inside the measured 0.18–0.28 m/s sim2real-friendly band.
    lin_vel_x: tuple[float, float] = (0.18, 0.28)
    legacy_lin_vel_x: tuple[float, float] = (0.45, 0.55)
    command_anneal_steps: int = 300 * 24
    lin_vel_y: tuple[float, float] = (0.0, 0.0)
    ang_vel_z: tuple[float, float] = (0.0, 0.0)
    zero_twist_probability: float = 0.10
    forward_only_steps: int = 300 * 24
    mixed_motion_steps: int = 700 * 24


@dataclass
class RollerControlCfg(ControlCfg):
    action_scale: float = 0.35
    action_filter_alpha: float = 0.35
    action_filter_warmup_steps: int = 2400
    # Human-like asymmetric actuator response: a push engages quickly, while
    # recovery/reversal settles slowly so one stroke is not cut into fragments.
    push_attack_alpha: float = 0.78
    recovery_release_alpha: float = 0.16
    minimum_push_hold_steps: int = 6
    # About 0.25 s at the 24 Hz control rate: long enough to finish one
    # stroke, short enough to let the opposite leg take over naturally.
    push_side_cooldown_steps: int = 12


@dataclass
class RollerInitStateCfg(InitStateCfg):
    root_height: float = 0.1385
    root_xy_noise: float = 0.002
    max_roll_deg: float = 2.0
    max_pitch_deg: float = 3.0
    joint_pos_noise: float = 0.01
    joint_vel_noise: float = 0.03


@dataclass
class RollerAssetCfg(AssetCfg):
    wheel_radius: float = 0.0175
    foot_names: tuple[str, ...] = (
        "left_front_wheel_collision",
        "left_rear_wheel_collision",
        "right_front_wheel_collision",
        "right_rear_wheel_collision",
    )


@dataclass
class RollerRewardCfg(RewardCfg):
    # Contact wheels sit near 0.02 m, so the previous clearance target
    # accidentally rewarded keeping the foot on the ground.  A higher target
    # makes a real swing phase advantageous.
    target_foot_height: float = 0.035
    foot_height_std: float = 0.015
    alternating_lift_weight: float = 0.0
    # Contact-side alternation is a complementary signal to action-side
    # pushes. Keep it modest so the duck may use a brief two-wheel transfer.
    support_switch_weight: float = 0.90
    rolling_consistency_weight: float = 2.0
    lateral_drift_weight: float = -2.0
    glide_efficiency_weight: float = 0.25
    survival_weight: float = 2.0
    wheel_propulsion_weight: float = 10.0
    # Keep the old shaping during the transition from existing checkpoints;
    # a later curriculum can ramp it down after the lower speed target settles.
    # Second-stage shaping: make the alternating push and quiet glide
    # visible without overpowering balance/velocity tracking.
    # Disabled in SmoothSkateV1: fixed clock phases encouraged timed leg flaps
    # even when the feet were not producing real propulsion.
    push_glide_phase_weight: float = 0.0
    glide_pose_weight: float = 0.50
    human_style_weight: float = 0.0
    smooth_action_weight: float = -0.45
    glide_event_weight: float = 1.5
    push_impulse_weight: float = 2.5
    # A bounded style term that closes the one-sided-push loophole without
    # forcing perfectly mirrored actions.
    alternating_push_weight: float = 1.25
    # Reward a complete per-leg stroke (push, lift, recontact) rather than
    # rewarding a leg merely for being the stronger side.
    stroke_cycle_weight: float = 1.50
    scales: dict[str, float] = field(
        default_factory=lambda: {
            **RewardCfg().scales,
            "track_linear_velocity": 2.0,
            "directional_velocity": 0.5,
            "track_angular_velocity": 1.10,
            "upright": 8.0,
            "pose": 1.0,
            "base_height": 2.0,
            # Make the contact pattern itself visible: reward a real swing
            # and one-foot transfer, while mildly charging persistent
            # two-foot contact.  Keep the values moderate for safe resume.
            "air_time": 0.80,
            "swing_time": 0.25,
            "single_support": 0.70,
            "double_support": -0.10,
            "foot_clearance": 0.80,
            "foot_slip": 0.0,
            "body_ang_vel": -0.20,
            "action_rate": -0.30,
            "joint_velocity": -0.004,
            "joint_limits": -5.0,
            "termination": -20.0,
            # Encourage a soft, alternating extend-push-coast rhythm instead
            # of holding both legs in one stiff pose.
            "push_glide_phase": 0.20,
            "glide_pose": 0.50,
        }
    )


@registry.envcfg(ENV_NAME)
@dataclass
class MicroduckRollersNpEnvCfg(MicroduckWalkNpEnvCfg):
    """First-stage straight passive-skating curriculum."""

    model_file: str = MODEL_FILE
    max_episode_seconds: float = 20.0
    commands: RollerCommandCfg = field(default_factory=RollerCommandCfg)
    control: RollerControlCfg = field(default_factory=RollerControlCfg)
    init_state: RollerInitStateCfg = field(default_factory=RollerInitStateCfg)
    asset: RollerAssetCfg = field(default_factory=RollerAssetCfg)
    reward: RollerRewardCfg = field(default_factory=RollerRewardCfg)

    def validate(self):
        super().validate()
        if not 0.0 < self.control.action_filter_alpha <= 1.0:
            raise ValueError("action_filter_alpha must be in (0, 1]")
