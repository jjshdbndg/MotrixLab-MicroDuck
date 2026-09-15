import numpy as np

from motrix_envs.locomotion.microduck.rollers_rewards import (
    alternating_push_score,
    forward_progress,
    glide_event_score,
    lateral_drift_cost,
    rolling_consistency,
    smooth_action_cost,
    stroke_cycle_transition,
)


def test_roller_reward_ordering():
    command = np.array([0.2, 0.2], dtype=np.float32)
    velocity = np.array([0.2, 0.0], dtype=np.float32)
    reward = forward_progress(command, velocity, std=0.08)
    assert reward[0] > reward[1]

    cost = lateral_drift_cost(np.array([0.0, 0.1], dtype=np.float32))
    np.testing.assert_allclose(cost, np.array([0.0, 0.01], dtype=np.float32))

    good = rolling_consistency(np.array([0.2]), np.array([10.0]), wheel_radius=0.02)
    bad = rolling_consistency(np.array([0.2]), np.array([0.0]), wheel_radius=0.02)
    assert good[0] > bad[0]


def test_smooth_action_cost_prefers_low_jerk():
    cost = smooth_action_cost(
        np.array([[0.1, 0.1]], dtype=np.float32),
        np.array([[0.1, 0.1]], dtype=np.float32),
        np.array([[0.1, 0.1]], dtype=np.float32),
    )
    spike = smooth_action_cost(
        np.array([[1.0, -1.0]], dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
    )
    assert cost[0] < spike[0]


def test_glide_event_requires_forward_motion_and_single_support():
    good = glide_event_score(
        np.array([0.18], dtype=np.float32),
        np.array([[True, False]]),
        np.array([0.05], dtype=np.float32),
        np.array([0.25], dtype=np.float32),
    )
    swizzle = glide_event_score(
        np.array([0.18], dtype=np.float32),
        np.array([[True, True]]),
        np.array([0.05], dtype=np.float32),
        np.array([0.25], dtype=np.float32),
    )
    assert good[0] > swizzle[0]


def test_alternating_push_prefers_switching_sides():
    switched = alternating_push_score(
        np.array([1], dtype=np.int8),
        np.array([0], dtype=np.int8),
        np.array([0], dtype=np.int32),
        np.array([0.18], dtype=np.float32),
        np.array([0.8], dtype=np.float32),
    )
    repeated = alternating_push_score(
        np.array([0], dtype=np.int8),
        np.array([0], dtype=np.int8),
        np.array([0], dtype=np.int32),
        np.array([0.18], dtype=np.float32),
        np.array([0.8], dtype=np.float32),
    )
    idle = alternating_push_score(
        np.array([-1], dtype=np.int8),
        np.array([0], dtype=np.int8),
        np.array([0.0], dtype=np.float32),
        np.array([0.18], dtype=np.float32),
        np.array([0.8], dtype=np.float32),
    )
    assert switched[0] > repeated[0]
    assert idle[0] == 0.0


def test_alternating_push_boosts_the_underrepresented_side():
    left = alternating_push_score(
        np.array([0], dtype=np.int8),
        np.array([1], dtype=np.int8),
        np.array([0], dtype=np.int32),
        np.array([0.18], dtype=np.float32),
        np.array([0.8], dtype=np.float32),
        left_push_count=np.array([1], dtype=np.int32),
        right_push_count=np.array([4], dtype=np.int32),
    )
    right = alternating_push_score(
        np.array([1], dtype=np.int8),
        np.array([1], dtype=np.int8),
        np.array([0], dtype=np.int32),
        np.array([0.18], dtype=np.float32),
        np.array([0.8], dtype=np.float32),
        left_push_count=np.array([1], dtype=np.int32),
        right_push_count=np.array([4], dtype=np.int32),
    )
    assert left[0] > right[0]


def test_stroke_cycle_transition_requires_push_lift_and_recontact():
    state, push, lift, complete = stroke_cycle_transition(
        np.array([0], dtype=np.int8),
        np.array([True]),
        np.array([True]),
        np.array([0.8], dtype=np.float32),
        np.array([0.2], dtype=np.float32),
    )
    assert state[0] == 1 and push[0] == 1.0 and lift[0] == 0.0 and complete[0] == 0.0
    state, push, lift, complete = stroke_cycle_transition(
        state,
        np.array([False]),
        np.array([True]),
        np.array([0.1], dtype=np.float32),
        np.array([0.2], dtype=np.float32),
    )
    assert state[0] == 2 and lift[0] == 1.0
    state, push, lift, complete = stroke_cycle_transition(
        state,
        np.array([True]),
        np.array([False]),
        np.array([0.1], dtype=np.float32),
        np.array([0.2], dtype=np.float32),
    )
    assert state[0] == 0 and complete[0] == 1.0
