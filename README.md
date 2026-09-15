# MotrixLab · MicroDuck Passive Rollers

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Simulation](https://img.shields.io/badge/Simulation-MotrixSim-6C5CE7)
![License](https://img.shields.io/badge/license-Apache--2.0- green)

An experimental MotrixLab environment where a MicroDuck learns to skate using passive rollers. The task focuses on alternating push-off cycles, balance, heading control, and smooth recovery after contact.

**Environment:** `microduck-flat-tile-passive-rollers`

**中文文档:** [MICRODUCK_ROLLERS.zh-CN.md](MICRODUCK_ROLLERS.zh-CN.md)

## What is included

| Area | Contents |
| --- | --- |
| Environment | Roller robot configuration, scene XML, observations and rewards |
| Training | RSLRL/PPO integration and resume helper |
| Evaluation | Deterministic evaluation script and focused regression tests |
| Checkpoint | `checkpoints/microduck_rollers/balanced_model_11600.pt` |
| Design notes | [Implementation spec](docs/superpowers/specs/2026-09-09-microduck-passive-roller-design.md) |

## Quick start

```bash
git clone https://github.com/jjshdbndg/MotrixLab-MicroDuck.git
cd MotrixLab-MicroDuck
git lfs pull
uv sync --all-packages --extra rslrl
```

Run the focused tests:

```bash
uv run pytest -q test/test_microduck_rollers_env.py \
  test/test_microduck_rollers_rewards.py \
  test/test_microduck_rollers_model.py
```

Evaluate the bundled checkpoint:

```bash
uv run python scripts/eval_microduck_rollers.py \
  --checkpoint checkpoints/microduck_rollers/balanced_model_11600.pt
```

For training and resume examples, see [scripts/resume_microduck_rollers.py](scripts/resume_microduck_rollers.py) and the Chinese guide.

## Repository map

```text
motrix_envs/   simulation environment and reward design
motrix_rl/     PPO/RSLRL task integration
scripts/       evaluation and resume entry points
test/          focused environment, model and reward tests
checkpoints/   documented example checkpoint
docs/          design notes
```

## Current status

- ✅ Passive-roller scene and alternating push-off reward cycle implemented
- ✅ Checkpoint, deterministic evaluation and regression tests included
- 🚧 Training quality and long-horizon skating stability are still under active iteration

## Relationship to MotrixLab

This repository is a focused extension of [MotrixLab](https://github.com/Motphys/MotrixLab), built on [MotrixSim](https://github.com/Motphys/motrixsim-docs). The changes here are intended to be easy to review, copy upstream, or use as an experiment branch.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) in the upstream project for details.
