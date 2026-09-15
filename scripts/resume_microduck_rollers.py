"""Continue passive-roller PPO training from a stable checkpoint."""

import argparse

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from motrix_envs import registry
from motrix_rl.registry import default_rl_cfg
from motrix_rl.rslrl.torch.wrap_vec_env import RslrlNpEnvWrap
from motrix_rl.skrl import get_log_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--start-iteration", type=int, default=2496)
    parser.add_argument(
        "--curriculum-iteration",
        type=int,
        default=0,
        help="SmoothSkate curriculum position; keep 0 when starting a new smooth run.",
    )
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = "microduck-flat-tile-passive-rollers"
    env = registry.make(env_name, "np", num_envs=args.num_envs)
    # PPO weights may resume from a late checkpoint, but the smooth-skate
    # curriculum must restart at stage A instead of jumping to high-speed
    # commands that recreate the old choppy gait.
    env.set_training_iteration(args.curriculum_iteration, 24)
    wrapped = RslrlNpEnvWrap(env, device)
    cfg = default_rl_cfg(env_name, "rslrl", "torch").runner.to_dict()
    cfg["save_interval"] = 20
    cfg["algorithm"]["learning_rate"] = args.learning_rate
    cfg["algorithm"]["entropy_coef"] = 0.002
    runner = OnPolicyRunner(
        wrapped,
        cfg,
        log_dir=get_log_dir(env_name, rllib="rslrl", agent_name="SmoothSkateV1"),
        device=device,
    )
    runner.load(args.checkpoint)
    runner.learn(num_learning_iterations=args.iterations)


if __name__ == "__main__":
    main()
