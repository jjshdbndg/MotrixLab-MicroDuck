# MicroDuck passive-roller checkpoint

`balanced_model_11600.pt` is the selected SmoothSkateV1 checkpoint. It is the best overall baseline among six retained candidates under the same deterministic evaluation, with the lowest measured heading drift while preserving balanced push cycles and smooth actions.

- Environment: `microduck-flat-tile-passive-rollers`
- Framework: RSL-RL PPO / PyTorch
- Simulation backend: MotrixSim NumPy
- Actor observation/action dimensions: 61 / 14
- Selection seed: 1234
- Visual rollout curriculum iteration: 1000
- SHA-256: `db795ccc12e0f60d649382c417fc01ae07de425c13dca6a6a53ff29e6b1f95f9`

The checkpoint includes the policy, value model, optimizer state, and observation normalizer required by `OnPolicyRunner.load()`. It is a work-in-progress baseline: the policy is stable but still curves instead of holding a true straight line. See [`MICRODUCK_ROLLERS.zh-CN.md`](../../MICRODUCK_ROLLERS.zh-CN.md) for playback, evaluation, and current limitations.
