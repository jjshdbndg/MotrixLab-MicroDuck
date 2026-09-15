"""Deterministic batch evaluation for a passive-roller checkpoint."""

import argparse

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from motrix_envs import registry
from motrix_rl.registry import default_rl_cfg
from motrix_rl.rslrl.torch.wrap_vec_env import RslrlNpEnvWrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=1000,
        help="Curriculum iteration used for commands; 1000 matches normal playback.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    # The NumPy environment owns reset and command randomization. Seed it so
    # checkpoints are compared on identical episodes.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = registry.make("microduck-flat-tile-passive-rollers", "np", num_envs=args.num_envs)
    env.cfg.noise.enabled = False
    env.set_training_iteration(args.start_iteration, 24)
    wrapped = RslrlNpEnvWrap(env, device)
    runner_cfg = default_rl_cfg(
        "microduck-flat-tile-passive-rollers", "rslrl", "torch"
    ).runner.to_dict()
    runner = OnPolicyRunner(wrapped, runner_cfg, log_dir=None, device=device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)

    obs, _ = wrapped.reset()
    alive = np.ones(args.num_envs, dtype=bool)
    survival_steps = np.zeros(args.num_envs, dtype=np.int32)
    distance = np.zeros(args.num_envs, dtype=np.float32)
    smooth_cost_sum = np.zeros(args.num_envs, dtype=np.float32)
    single_support_sum = np.zeros(args.num_envs, dtype=np.float32)
    left_push_sum = np.zeros(args.num_envs, dtype=np.float32)
    right_push_sum = np.zeros(args.num_envs, dtype=np.float32)
    push_switch_sum = np.zeros(args.num_envs, dtype=np.float32)
    push_balance_sum = np.zeros(args.num_envs, dtype=np.float32)
    left_lift_sum = np.zeros(args.num_envs, dtype=np.float32)
    right_lift_sum = np.zeros(args.num_envs, dtype=np.float32)
    left_complete_sum = np.zeros(args.num_envs, dtype=np.float32)
    right_complete_sum = np.zeros(args.num_envs, dtype=np.float32)
    abs_yaw_rate_sum = np.zeros(args.num_envs, dtype=np.float32)
    heading_change = np.zeros(args.num_envs, dtype=np.float32)
    samples = np.zeros(args.num_envs, dtype=np.int32)
    success_at_10s = None
    commanded_speed = env.state.info["commands"][:, 0].copy()
    dt = env.cfg.ctrl_dt
    ten_second_step = min(env.cfg.max_episode_steps, max(1, round(10.0 / dt)))

    with torch.inference_mode():
        for step in range(env.cfg.max_episode_steps):
            active = alive.copy()
            obs, _, _, _ = wrapped.step(policy(obs))
            physical_done = env.state.terminated.copy()
            velocity = env.get_local_linvel(env.state.data)[:, 0]
            yaw_rate = env.get_gyro(env.state.data)[:, 2]
            smooth_cost_sum[active] += env.state.info["metrics"]["action_smoothness_cost"][active]
            single_support_sum[active] += env.state.info["metrics"]["single_support_fraction"][active]
            left_push_sum[active] = env.state.info["metrics"]["left_push_fraction"][active]
            right_push_sum[active] = env.state.info["metrics"]["right_push_fraction"][active]
            push_switch_sum[active] = env.state.info["metrics"]["push_switch_rate"][active]
            push_balance_sum[active] = env.state.info["metrics"]["push_side_balance"][active]
            left_lift_sum[active] = env.state.info["metrics"]["left_lift_fraction"][active]
            right_lift_sum[active] = env.state.info["metrics"]["right_lift_fraction"][active]
            left_complete_sum[active] = env.state.info["metrics"]["left_complete_fraction"][active]
            right_complete_sum[active] = env.state.info["metrics"]["right_complete_fraction"][active]
            abs_yaw_rate_sum[active] += np.abs(yaw_rate[active])
            heading_change[active] += yaw_rate[active] * dt
            samples[active] += 1
            survival_steps[active] += 1
            distance[active] += velocity[active] * dt
            alive &= ~physical_done
            if step + 1 == ten_second_step:
                success_at_10s = alive.copy()

    if success_at_10s is None:
        success_at_10s = alive.copy()
    print(f"success_10s={np.mean(success_at_10s):.3f}")
    print(f"success_full_episode={np.mean(alive):.3f}")
    print(f"mean_survival_s={np.mean(survival_steps) * dt:.3f}")
    print(f"mean_command_mps={np.mean(commanded_speed):.3f}")
    print(f"mean_distance_m={np.mean(distance):.3f}")
    print(f"successful_mean_distance_m={np.mean(distance[alive]) if np.any(alive) else 0.0:.3f}")
    smooth = smooth_cost_sum / np.maximum(samples, 1)
    support = single_support_sum / np.maximum(samples, 1)
    # Higher is better; the exponential keeps the score readable and bounded.
    smoothness_score = np.exp(-8.0 * smooth) * np.clip(support / 0.25, 0.0, 1.0)
    print(f"mean_action_smoothness_cost={np.mean(smooth):.6f}")
    print(f"mean_single_support_fraction={np.mean(support):.3f}")
    print(f"mean_smoothness_score={np.mean(smoothness_score):.3f}")
    print(f"mean_left_push_fraction={np.mean(left_push_sum):.3f}")
    print(f"mean_right_push_fraction={np.mean(right_push_sum):.3f}")
    print(f"mean_push_switch_rate={np.mean(push_switch_sum):.3f}")
    print(f"mean_push_side_balance={np.mean(push_balance_sum):.3f}")
    print(f"mean_left_lift_fraction={np.mean(left_lift_sum):.3f}")
    print(f"mean_right_lift_fraction={np.mean(right_lift_sum):.3f}")
    print(f"mean_left_complete_fraction={np.mean(left_complete_sum):.3f}")
    print(f"mean_right_complete_fraction={np.mean(right_complete_sum):.3f}")
    mean_abs_yaw_rate = abs_yaw_rate_sum / np.maximum(samples, 1)
    print(f"mean_abs_yaw_rate_rad_s={np.mean(mean_abs_yaw_rate):.4f}")
    print(f"mean_net_heading_change_deg={np.degrees(np.mean(heading_change)):.2f}")
    curvature = np.abs(heading_change) / np.maximum(np.abs(distance), 0.1)
    print(f"mean_heading_change_per_meter_rad_m={np.mean(curvature):.4f}")


if __name__ == "__main__":
    main()
