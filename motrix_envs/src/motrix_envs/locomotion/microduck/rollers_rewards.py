"""Finite, vectorized reward helpers for passive skating."""

import numpy as np


def _finite(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(value)):
        raise FloatingPointError(f"Non-finite roller reward term: {name}")
    return value


def forward_progress(command: np.ndarray, velocity: np.ndarray, std: float = 0.08) -> np.ndarray:
    value = np.exp(-np.square(np.asarray(command) - np.asarray(velocity)) / std**2)
    return _finite(value, "forward_progress")


def lateral_drift_cost(lateral_velocity: np.ndarray) -> np.ndarray:
    return _finite(np.square(lateral_velocity), "lateral_drift")


def rolling_consistency(
    forward_velocity: np.ndarray,
    wheel_angular_velocity: np.ndarray,
    wheel_radius: float,
    std: float = 0.08,
) -> np.ndarray:
    rolling_velocity = np.abs(wheel_angular_velocity) * wheel_radius
    error = np.abs(forward_velocity) - rolling_velocity
    return _finite(np.exp(-np.square(error) / std**2), "rolling_consistency")


def glide_efficiency(
    forward_velocity: np.ndarray,
    action_rate: np.ndarray,
    target_speed: float = 0.2,
) -> np.ndarray:
    coast = np.exp(-np.square(action_rate) / 0.25**2)
    progress = np.clip(forward_velocity / max(target_speed, 1.0e-6), 0.0, 1.0)
    return _finite(coast * progress, "glide_efficiency")


def smooth_action_cost(
    actions: np.ndarray,
    previous_actions: np.ndarray,
    previous_previous_actions: np.ndarray,
) -> np.ndarray:
    """Squared action acceleration (discrete jerk proxy), per environment."""
    a = np.asarray(actions, dtype=np.float32)
    p = np.asarray(previous_actions, dtype=np.float32)
    pp = np.asarray(previous_previous_actions, dtype=np.float32)
    acceleration = a - 2.0 * p + pp
    return _finite(np.mean(np.square(acceleration), axis=-1), "smooth_action")


def glide_event_score(
    forward_velocity: np.ndarray,
    contacts: np.ndarray,
    action_rate: np.ndarray,
    glide_seconds: np.ndarray,
    min_glide_seconds: float = 0.20,
    max_glide_seconds: float = 0.45,
) -> np.ndarray:
    """Reward real forward single-support glides, not two-foot swizzles."""
    velocity = np.asarray(forward_velocity, dtype=np.float32)
    contact_mask = np.asarray(contacts, dtype=bool)
    rate = np.asarray(action_rate, dtype=np.float32)
    duration = np.asarray(glide_seconds, dtype=np.float32)
    single = (np.sum(contact_mask, axis=1) == 1).astype(np.float32)
    in_window = (
        (duration >= min_glide_seconds) & (duration <= max_glide_seconds)
    ).astype(np.float32)
    forward_gate = np.clip(velocity / 0.18, 0.0, 1.0)
    stillness = np.exp(-np.square(rate) / 0.20**2)
    return _finite(single * in_window * forward_gate * stillness, "glide_event")


def alternating_push_score(
    current_side: np.ndarray,
    previous_side: np.ndarray,
    cooldown_steps: np.ndarray,
    forward_velocity: np.ndarray,
    drive_strength: np.ndarray,
    min_drive: float = 0.25,
    left_push_count: np.ndarray | None = None,
    right_push_count: np.ndarray | None = None,
) -> np.ndarray:
    """Reward a real change of the leg that is doing the push.

    ``current_side`` and ``previous_side`` use ``0`` for left, ``1`` for
    right, and ``-1`` for no detected push.  The cooldown is deliberately
    part of this helper so a single stroke cannot earn a switch reward on
    every simulation step.  A small charge on repeated same-side strokes
    prevents the policy from settling into a permanent one-foot solution,
    while the first detected stroke remains neutral (there is no side to
    alternate from yet).
    """
    side = np.asarray(current_side, dtype=np.int8)
    previous = np.asarray(previous_side, dtype=np.int8)
    cooldown = np.asarray(cooldown_steps, dtype=np.int32)
    velocity = np.asarray(forward_velocity, dtype=np.float32)
    drive = np.asarray(drive_strength, dtype=np.float32)
    if not (side.shape == previous.shape == cooldown.shape == velocity.shape == drive.shape):
        raise ValueError("alternating_push_score inputs must have the same shape")

    valid = (side >= 0) & (drive >= min_drive) & (cooldown <= 0)
    has_previous = previous >= 0
    gate = np.clip((velocity - 0.10) / 0.10, 0.0, 1.0)
    switched = valid & has_previous & (side != previous)
    repeated = valid & has_previous & (side == previous)
    # Make a genuine side change clearly more valuable than continuing to
    # push from the same side, while keeping the charge small enough that the
    # policy can finish a stroke during the cooldown window.
    score = gate * (1.10 * switched.astype(np.float32) - 0.24 * repeated.astype(np.float32))
    if left_push_count is not None or right_push_count is not None:
        if left_push_count is None or right_push_count is None:
            raise ValueError("left_push_count and right_push_count must be provided together")
        left_count = np.asarray(left_push_count, dtype=np.float32)
        right_count = np.asarray(right_push_count, dtype=np.float32)
        if left_count.shape != side.shape or right_count.shape != side.shape:
            raise ValueError("push count inputs must match current_side shape")
        # Give the currently underrepresented side a bounded bonus.  This is
        # an episode-local fairness signal, not a fixed left/right preference.
        balance = np.clip(
            (right_count - left_count) / (left_count + right_count + 1.0),
            -1.0,
            1.0,
        )
        side_bias = np.where(side == 0, balance, np.where(side == 1, -balance, 0.0))
        score += gate * 0.55 * side_bias * valid.astype(np.float32)
    return _finite(score, "alternating_push")


def stroke_cycle_transition(
    cycle_state: np.ndarray,
    contact: np.ndarray,
    previous_contact: np.ndarray,
    drive_strength: np.ndarray,
    min_drive: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance one leg through push -> lift -> recontact.

    State values are ``0`` ready/contact, ``1`` pushing and ``2`` airborne.
    Events are emitted only on the edge that causes the transition, so a
    sustained contact or airborne period cannot repeatedly earn reward.
    """
    state = np.asarray(cycle_state, dtype=np.int8)
    current = np.asarray(contact, dtype=bool)
    previous = np.asarray(previous_contact, dtype=bool)
    drive = np.asarray(drive_strength, dtype=np.float32)
    if not (state.shape == current.shape == previous.shape == drive.shape):
        raise ValueError("stroke_cycle_transition inputs must have the same shape")
    next_state = state.copy()
    push = np.zeros(state.shape, dtype=np.float32)
    lift = np.zeros(state.shape, dtype=np.float32)
    complete = np.zeros(state.shape, dtype=np.float32)

    start_push = (state == 0) & current & (drive >= min_drive)
    next_state[start_push] = 1
    push[start_push] = 1.0

    lifted = (state == 1) & previous & ~current
    next_state[lifted] = 2
    lift[lifted] = 1.0

    landed = (state == 2) & ~previous & current
    next_state[landed] = 0
    complete[landed] = 1.0
    return next_state, push, lift, complete


def push_glide_phase_score(
    left_drive: np.ndarray,
    right_drive: np.ndarray,
    phase: np.ndarray,
    action_rate: np.ndarray,
) -> np.ndarray:
    """Reward alternating single-leg push phases and quiet glide phases.

    ``phase`` runs from 0 to 1 over one skating cycle.  The first and third
    quarters are left/right push windows; the intervening windows are glide.
    The score is deliberately bounded so it shapes style without overpowering
    balance and velocity objectives.
    """
    left_drive = np.asarray(left_drive, dtype=np.float32)
    right_drive = np.asarray(right_drive, dtype=np.float32)
    phase = np.asarray(phase, dtype=np.float32)
    action_rate = np.asarray(action_rate, dtype=np.float32)
    left_push = ((phase >= 0.05) & (phase < 0.30)).astype(np.float32)
    right_push = ((phase >= 0.55) & (phase < 0.80)).astype(np.float32)
    glide = 1.0 - np.maximum(left_push, right_push)
    left_advantage = np.tanh(4.0 * (left_drive - right_drive))
    right_advantage = np.tanh(4.0 * (right_drive - left_drive))
    push_score = 0.5 * (left_push * left_advantage + right_push * right_advantage)
    glide_score = glide * np.exp(-np.square(action_rate) / 0.20**2)
    return _finite(push_score + 0.5 * glide_score, "push_glide_phase")


def human_skate_style_score(
    left_sagittal: np.ndarray,
    right_sagittal: np.ndarray,
    left_lateral: np.ndarray,
    right_lateral: np.ndarray,
    phase: np.ndarray,
    action_rate: np.ndarray,
) -> np.ndarray:
    """Shape a human-like extend, push, transfer and glide cycle.

    Sagittal commands are signed; lateral commands are penalized so the policy
    cannot gain speed by shuffling side-to-side.  Each push window rewards one
    leg being active while the other is quiet, and glide windows reward both
    legs returning toward neutral with low command-rate.
    """
    l = np.asarray(left_sagittal, dtype=np.float32)
    r = np.asarray(right_sagittal, dtype=np.float32)
    ll = np.asarray(left_lateral, dtype=np.float32)
    rl = np.asarray(right_lateral, dtype=np.float32)
    p = np.asarray(phase, dtype=np.float32)
    rate = np.asarray(action_rate, dtype=np.float32)
    left_window = ((p >= 0.08) & (p < 0.34)).astype(np.float32)
    right_window = ((p >= 0.58) & (p < 0.84)).astype(np.float32)
    glide_window = 1.0 - np.maximum(left_window, right_window)
    l_drive = np.tanh(3.0 * np.abs(l))
    r_drive = np.tanh(3.0 * np.abs(r))
    alternation = left_window * l_drive * (1.0 - r_drive) + right_window * r_drive * (1.0 - l_drive)
    extension = 0.5 * (left_window * np.maximum(l, 0.0) + right_window * np.maximum(r, 0.0))
    quiet_glide = glide_window * np.exp(-np.square(rate) / 0.16**2) * np.exp(-(l*l + r*r) / 0.35**2)
    lateral_penalty = 0.35 * (ll * ll + rl * rl)
    return _finite(alternation + extension + 0.5 * quiet_glide - lateral_penalty, "human_skate_style")
