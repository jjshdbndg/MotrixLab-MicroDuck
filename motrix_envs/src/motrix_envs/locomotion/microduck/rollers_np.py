"""MicroDuck passive-roller environment."""

import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.locomotion.microduck.rollers_cfg import ENV_NAME, MicroduckRollersNpEnvCfg
from motrix_envs.locomotion.microduck.rollers_rewards import (
    alternating_push_score,
    glide_efficiency,
    glide_event_score,
    human_skate_style_score,
    lateral_drift_cost,
    push_glide_phase_score,
    rolling_consistency,
    smooth_action_cost,
    stroke_cycle_transition,
)
from motrix_envs.locomotion.microduck.walk_np import MicroduckWalkTask


@registry.env(ENV_NAME, sim_backend="np")
class MicroduckRollersTask(MicroduckWalkTask):
    """Straight skating driven only by leg servos and center-of-mass control."""

    def __init__(self, cfg: MicroduckRollersNpEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs)
        joint_names = tuple(self._model.joint_names)
        self._wheel_joint_indices = np.asarray(
            [joint_names.index(name) for name in (
                "passive_LF_wheel",
                "passive_LR_wheel",
                "passive_RF_wheel",
                "passive_RR_wheel",
            )],
            dtype=np.int64,
        )
        self._smooth_filter_steps = 0
        self._push_hold_remaining = np.zeros(num_envs, dtype=np.int32)

    def reset(self, data):
        obs, info = super().reset(data)
        n = data.shape[0]
        info["raw_actions"] = np.zeros((n, 14), dtype=np.float32)
        info["previous_actions"] = np.zeros((n, 14), dtype=np.float32)
        info["previous_previous_actions"] = np.zeros((n, 14), dtype=np.float32)
        info["single_support_time"] = np.zeros(n, dtype=np.float32)
        info["previous_wheel_vel"] = np.zeros((n, 4), dtype=np.float32)
        # Push-side memory is intentionally per episode.  A side is only
        # updated after a meaningful, forward-moving stroke and then held for
        # a short cooldown so action noise cannot manufacture rapid switches.
        info["previous_push_side"] = np.full(n, -1, dtype=np.int8)
        info["push_side_cooldown"] = np.zeros(n, dtype=np.int32)
        info["left_push_steps"] = np.zeros(n, dtype=np.int32)
        info["right_push_steps"] = np.zeros(n, dtype=np.int32)
        info["push_switches"] = np.zeros(n, dtype=np.int32)
        info["push_events"] = np.zeros(n, dtype=np.int32)
        info["previous_contacts"] = np.zeros((n, 2), dtype=bool)
        info["stroke_cycle_state"] = np.zeros((n, 2), dtype=np.int8)
        info["stroke_push_events"] = np.zeros((n, 2), dtype=np.int32)
        info["stroke_lift_events"] = np.zeros((n, 2), dtype=np.int32)
        info["stroke_complete_events"] = np.zeros((n, 2), dtype=np.int32)
        return obs, info

    def _sample_commands(self, num_envs: int) -> np.ndarray:
        commands = np.zeros((num_envs, 13), dtype=np.float32)
        stage = self._command_stage()
        if stage == 0:
            speed_range = (0.06, 0.10)
        elif stage == 1:
            speed_range = (0.08, 0.12)
        else:
            # Smoothly move an existing policy from the old high-speed command
            # distribution to the stable passive-roller band. This keeps the
            # actor normalizer and gait latent state in-distribution while the
            # target is lowered over one curriculum window.
            start = self.cfg.commands.mixed_motion_steps
            span = max(1, self.cfg.commands.command_anneal_steps)
            alpha = np.clip((self._global_step - start) / span, 0.0, 1.0)
            old_lo, old_hi = self.cfg.commands.legacy_lin_vel_x
            new_lo, new_hi = self.cfg.commands.lin_vel_x
            speed_range = (
                (1.0 - alpha) * old_lo + alpha * new_lo,
                (1.0 - alpha) * old_hi + alpha * new_hi,
            )
        commands[:, 0] = self._uniform(speed_range, num_envs)
        idle = np.random.random(num_envs) < self.cfg.commands.zero_twist_probability
        commands[idle, 0] = 0.0
        return commands

    def apply_action(self, actions: np.ndarray, state):
        """Apply a light first-order target filter to remove high-frequency kicks."""
        raw = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        previous = state.info["current_actions"]
        warmup = max(1, self.cfg.control.action_filter_warmup_steps)
        progress = np.clip(self._smooth_filter_steps / warmup, 0.0, 1.0)
        # Start with the legacy dynamics when resuming, then gently introduce
        # the low-pass filter so a checkpoint is not destabilized on step one.
        base_alpha = 1.0 - progress * (1.0 - self.cfg.control.action_filter_alpha)
        self._smooth_filter_steps += 1
        attack = max(base_alpha, self.cfg.control.push_attack_alpha)
        release = min(base_alpha, self.cfg.control.recovery_release_alpha)
        alpha = np.where(np.abs(raw) >= np.abs(previous), attack, release)
        filtered = previous + alpha.astype(np.float32) * (raw - previous)
        hold = self._push_hold_remaining > 0
        filtered[hold] = previous[hold] + 0.35 * (filtered[hold] - previous[hold])
        self._push_hold_remaining[hold] -= 1
        trigger = np.max(np.abs(filtered - previous), axis=1) > 0.18
        self._push_hold_remaining[(~hold) & trigger] = self.cfg.control.minimum_push_hold_steps
        state.info["raw_actions"] = raw
        return super().apply_action(filtered, state)

    def get_wheel_vel(self, data: mtx.SceneData) -> np.ndarray:
        return self._body.get_joint_dof_vel(data)[:, self._wheel_joint_indices]

    def update_observation(self, state):
        previous_contacts = state.info["contacts"].copy()
        state.info["previous_contacts"] = previous_contacts
        previous_memory = state.info.get(
            "support_memory_side", np.full(self.num_envs, -1, dtype=np.int8)
        ).copy()
        previous_memory_valid = previous_memory >= 0
        wheel_contacts = self._model.get_contact_query(state.data).is_colliding(self._foot_contact_pairs)
        wheel_contacts = wheel_contacts.reshape((self.num_envs, 4))
        contacts = np.column_stack((
            np.any(wheel_contacts[:, :2], axis=1),
            np.any(wheel_contacts[:, 2:], axis=1),
        ))
        foot_positions = np.stack([foot.get_position(state.data) for foot in self._feet], axis=1)
        state.info["foot_velocity"] = (foot_positions - state.info["foot_positions"]) / self.cfg.ctrl_dt
        state.info["foot_positions"] = foot_positions
        air_time = state.info["feet_air_time"] + self.cfg.ctrl_dt
        first_contact = np.logical_and(contacts, np.logical_not(state.info["contacts"]))
        state.info["landing_air_time"] = np.where(first_contact, air_time, 0.0)
        state.info["feet_air_time"] = np.where(contacts, 0.0, air_time)
        state.info["swing_accum"] += np.logical_not(contacts) * self.cfg.ctrl_dt
        state.info["contacts"] = contacts
        single_support = np.sum(contacts, axis=1) == 1
        state.info["single_support_time"] = np.where(
            single_support, state.info["single_support_time"] + self.cfg.ctrl_dt, 0.0
        )
        current_single = np.sum(contacts, axis=1) == 1
        # Side encoding: 0 = left support, 1 = right support.
        current_side = (~contacts[:, 0]).astype(np.int8)
        previous_support = np.zeros((self.num_envs, 2), dtype=np.float32)
        previous_support[:, 0] = previous_memory_valid & (previous_memory == 0)
        previous_support[:, 1] = previous_memory_valid & (previous_memory == 1)
        state.info["previous_support_side"] = previous_support
        expected_side = 1 - previous_memory
        alternation = (
            current_single
            & previous_memory_valid
            & (current_side == expected_side)
        ).astype(np.float32)
        same_side = (
            current_single
            & previous_memory_valid
            & (current_side == previous_memory)
        ).astype(np.float32)
        state.info["support_switch"] = (
            alternation
        ).astype(np.float32)
        state.info["support_alternation"] = alternation - same_side
        updated_memory = previous_memory.copy()
        updated_memory[current_single] = current_side[current_single]
        state.info["support_memory_side"] = updated_memory
        state.info["wheel_contacts"] = wheel_contacts
        return state.replace(obs=self._get_obs(state.data, state.info))

    def update_reward(self, state):
        state = super().update_reward(state)
        wheel_vel = self.get_wheel_vel(state.data)
        local_velocity = self.get_local_linvel(state.data)
        forward_velocity = local_velocity[:, 0]
        contacts = state.info["contacts"]
        rolling_speed = np.mean(np.abs(wheel_vel), axis=1) * self.cfg.asset.wheel_radius
        action_rate = np.mean(np.abs(state.info["current_actions"] - state.info["last_actions"]), axis=1)
        command_speed = np.maximum(state.info["commands"][:, 0], 0.05)
        speed_gate = np.clip(np.abs(forward_velocity) / command_speed, 0.0, 1.0)
        stage = self._command_stage()
        survival_weight = self.cfg.reward.survival_weight if stage == 0 else 1.5 if stage == 1 else 0.5
        balance_gate = np.clip(
            (-self._projected_gravity(state.data)[:, 2] - 0.45) / 0.45,
            0.0,
            1.0,
        )
        # Match the official roller task: forward wheel spin is the discovery
        # signal. Body-speed tracking then converts that spin into useful travel.
        forward_omega = np.mean(wheel_vel, axis=1)
        previous_wheel_vel = state.info.get("previous_wheel_vel", wheel_vel)
        wheel_accel = np.maximum(forward_omega - np.mean(previous_wheel_vel, axis=1), 0.0)
        omega_scale = 0.3 / self.cfg.asset.wheel_radius
        propulsion = (
            np.maximum(state.info["commands"][:, 0], 0.0)
            * np.tanh(np.maximum(forward_omega, 0.0) / omega_scale)
            * balance_gate
        )
        roller_terms = {
            "rolling_consistency": rolling_consistency(
                forward_velocity, np.mean(np.abs(wheel_vel), axis=1), self.cfg.asset.wheel_radius
            ) * speed_gate * self.cfg.reward.rolling_consistency_weight,
            "lateral_drift": lateral_drift_cost(local_velocity[:, 1]) * self.cfg.reward.lateral_drift_weight,
            "glide_efficiency": glide_efficiency(forward_velocity, action_rate)
            * self.cfg.reward.glide_efficiency_weight,
            "survival": np.logical_not(state.terminated).astype(np.float32) * survival_weight,
            "wheel_propulsion": propulsion.astype(np.float32) * self.cfg.reward.wheel_propulsion_weight,
            "smooth_action": smooth_action_cost(
                state.info["current_actions"],
                state.info["last_actions"],
                state.info["previous_actions"],
            ) * self.cfg.reward.smooth_action_weight,
            "glide_event": glide_event_score(
                forward_velocity,
                contacts,
                action_rate,
                state.info["single_support_time"],
            ) * self.cfg.reward.glide_event_weight,
            "push_impulse": np.tanh(wheel_accel / 8.0)
            * np.clip(forward_velocity / 0.12, 0.0, 1.0)
            * self.cfg.reward.push_impulse_weight,
        }
        # Style shaping: alternate a clear leg extension/push with a quiet
        # two-wheel glide.  Using the three sagittal leg servos makes this
        # compatible with the 14-D policy while leaving balance in charge.
        actions = state.info["current_actions"]
        left_drive = np.mean(np.abs(actions[:, [2, 3, 4]]), axis=1)
        right_drive = np.mean(np.abs(actions[:, [11, 12, 13]]), axis=1)
        drive_strengths = np.column_stack((left_drive, right_drive))
        cycle_state, stroke_push, stroke_lift, stroke_complete = stroke_cycle_transition(
            state.info["stroke_cycle_state"],
            contacts,
            state.info["previous_contacts"],
            drive_strengths,
        )
        state.info["stroke_cycle_state"] = cycle_state
        state.info["stroke_push_events"] += stroke_push.astype(np.int32)
        state.info["stroke_lift_events"] += stroke_lift.astype(np.int32)
        state.info["stroke_complete_events"] += stroke_complete.astype(np.int32)
        drive_strength = np.maximum(left_drive, right_drive)
        drive_margin = 0.10
        candidate_side = np.full(actions.shape[0], -1, dtype=np.int8)
        candidate_side[left_drive > right_drive + drive_margin] = 0
        candidate_side[right_drive > left_drive + drive_margin] = 1
        forward_gate = forward_velocity > 0.10
        valid_candidate = (candidate_side >= 0) & (drive_strength >= 0.25) & forward_gate
        cooldown = np.maximum(state.info["push_side_cooldown"] - 1, 0)
        previous_push_side = state.info["previous_push_side"]
        push_score = alternating_push_score(
            candidate_side,
            previous_push_side,
            cooldown,
            forward_velocity,
            drive_strength,
            left_push_count=state.info["left_push_steps"],
            right_push_count=state.info["right_push_steps"],
        )
        # Accept at most one push-side event per cooldown window.  The first
        # event establishes the side; subsequent events reward a true switch
        # and mildly charge a repeated same-side stroke.
        accepted = valid_candidate & (cooldown <= 0)
        switched = accepted & (previous_push_side >= 0) & (
            candidate_side != previous_push_side
        )
        state.info["push_switches"] += switched.astype(np.int32)
        state.info["push_events"] += accepted.astype(np.int32)
        state.info["left_push_steps"] += (accepted & (candidate_side == 0)).astype(np.int32)
        state.info["right_push_steps"] += (accepted & (candidate_side == 1)).astype(np.int32)
        updated_side = previous_push_side.copy()
        updated_side[accepted] = candidate_side[accepted]
        state.info["previous_push_side"] = updated_side
        cooldown[accepted] = self.cfg.control.push_side_cooldown_steps
        state.info["push_side_cooldown"] = cooldown
        phase = (state.info["steps"].astype(np.float32) * self.cfg.ctrl_dt / 1.2) % 1.0
        phase_score = push_glide_phase_score(left_drive, right_drive, phase, action_rate)
        roller_terms["push_glide_phase"] = (
            phase_score * self.cfg.reward.push_glide_phase_weight * balance_gate
        )
        roller_terms["glide_pose"] = (
            glide_efficiency(forward_velocity, action_rate)
            * self.cfg.reward.glide_pose_weight
            * balance_gate
        )
        human_style = human_skate_style_score(
            actions[:, 2],
            actions[:, 11],
            0.5 * (actions[:, 0] + actions[:, 1]),
            0.5 * (actions[:, 9] + actions[:, 10]),
            phase,
            action_rate,
        )
        roller_terms["human_style"] = human_style * self.cfg.reward.human_style_weight * balance_gate
        roller_terms["alternating_push"] = (
            push_score * self.cfg.reward.alternating_push_weight * balance_gate
        )
        stroke_events = (
            np.sum(stroke_complete, axis=1)
            + 0.20 * np.sum(stroke_lift, axis=1)
            + 0.05 * np.sum(stroke_push, axis=1)
        )
        roller_terms["stroke_cycle"] = (
            stroke_events
            * self.cfg.reward.stroke_cycle_weight
            * speed_gate
            * balance_gate
        )

        # The generic clearance term can be satisfied by a nearly grounded
        # roller.  Require a signed, phase-matched height difference instead:
        # left foot visibly higher in the left swing window, then vice versa.
        left_swing = np.logical_and(~contacts[:, 0], contacts[:, 1]).astype(np.float32)
        right_swing = np.logical_and(contacts[:, 0], ~contacts[:, 1]).astype(np.float32)
        swing_window = left_swing + right_swing
        desired_sign = left_swing - right_swing
        foot_height_delta = (
            state.info["foot_positions"][:, 0, 2]
            - state.info["foot_positions"][:, 1, 2]
        )
        alternating_lift = swing_window * np.tanh(
            (desired_sign * foot_height_delta - 0.008) / 0.012
        )
        roller_terms["alternating_lift"] = (
            alternating_lift
            * self.cfg.reward.alternating_lift_weight
            * speed_gate
            * balance_gate
        )
        roller_terms["support_switch"] = (
            state.info.get("support_alternation", np.zeros(self.num_envs, dtype=np.float32))
            * self.cfg.reward.support_switch_weight
            * balance_gate
        )
        state.info["Reward"].update(roller_terms)
        state.reward += np.sum(np.stack(tuple(roller_terms.values()), axis=0), axis=0).astype(np.float32)
        state.info["metrics"].update({
            "wheel_speed": rolling_speed,
            "rolling_error": np.abs(np.abs(forward_velocity) - rolling_speed),
            "wheel_contact_fraction": np.mean(state.info["wheel_contacts"], axis=1),
            "action_smoothness_cost": smooth_action_cost(
                state.info["current_actions"],
                state.info["last_actions"],
                state.info["previous_actions"],
            ),
            "single_support_time": state.info["single_support_time"],
            "left_push_fraction": state.info["left_push_steps"] / np.maximum(state.info["push_events"], 1),
            "right_push_fraction": state.info["right_push_steps"] / np.maximum(state.info["push_events"], 1),
            "push_switch_rate": state.info["push_switches"] / np.maximum(state.info["push_events"] - 1, 1),
            "push_side_balance": 1.0 - np.abs(
                state.info["left_push_steps"] - state.info["right_push_steps"]
            ) / np.maximum(state.info["push_events"], 1),
            "left_lift_fraction": state.info["stroke_lift_events"][:, 0]
            / np.maximum(state.info["stroke_push_events"][:, 0], 1),
            "right_lift_fraction": state.info["stroke_lift_events"][:, 1]
            / np.maximum(state.info["stroke_push_events"][:, 1], 1),
            "left_complete_fraction": state.info["stroke_complete_events"][:, 0]
            / np.maximum(state.info["stroke_push_events"][:, 0], 1),
            "right_complete_fraction": state.info["stroke_complete_events"][:, 1]
            / np.maximum(state.info["stroke_push_events"][:, 1], 1),
        })
        state.info["previous_previous_actions"] = state.info["previous_actions"].copy()
        state.info["previous_actions"] = state.info["last_actions"].copy()
        state.info["previous_wheel_vel"] = wheel_vel.copy()
        return state

    def _reward_terms(self, state):
        terms = super()._reward_terms(state)
        gravity = self._projected_gravity(state.data)
        height = self._body.get_pose(state.data)[:, 2]
        upright_gate = np.clip((-gravity[:, 2] - 0.45) / 0.45, 0.0, 1.0)
        height_gate = np.clip((height - self.cfg.termination.min_root_height) / 0.04, 0.0, 1.0)
        stage_scale = 1.0 if self._command_stage() == 0 else 4.0 if self._command_stage() == 1 else 8.0
        locomotion_gate = upright_gate * height_gate
        for name in (
            "track_linear_velocity",
            "directional_velocity",
            "air_time",
            "swing_time",
            "foot_clearance",
            "single_support",
        ):
            terms[name] *= locomotion_gate
        terms["track_linear_velocity"] *= stage_scale
        terms["directional_velocity"] *= stage_scale
        return terms
