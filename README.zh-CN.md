# MotrixLab · MicroDuck 被动轮轮滑

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Simulation](https://img.shields.io/badge/Simulation-MotrixSim-6C5CE7)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

这是一个 MotrixLab 实验分支：让 MicroDuck 双足机器人使用被动轮完成交替蹬滑。环境重点关注“蹬地—离地—回收—重新接触”的周期奖励、平衡、航向控制和接触后的平滑恢复。

**环境名：** `microduck-flat-tile-passive-rollers`

**English:** [README.md](README.md)

## 项目内容

| 模块 | 内容 |
| --- | --- |
| 仿真环境 | 轮滑机器人配置、场景 XML、观测与奖励函数 |
| 训练接入 | RSLRL/PPO 任务接入与续训脚本 |
| 评估验证 | 确定性评估脚本与专项回归测试 |
| 示例模型 | `checkpoints/microduck_rollers/balanced_model_11600.pt` |
| 设计说明 | [实现方案](docs/superpowers/specs/2026-09-09-microduck-passive-roller-design.md) |

## 快速开始

```bash
git clone https://github.com/jjshdbndg/MotrixLab-MicroDuck.git
cd MotrixLab-MicroDuck
git lfs pull
uv sync --all-packages --extra rslrl
```

运行专项测试：

```bash
uv run pytest -q test/test_microduck_rollers_env.py \
  test/test_microduck_rollers_rewards.py \
  test/test_microduck_rollers_model.py
```

评估仓库内置模型：

```bash
uv run python scripts/eval_microduck_rollers.py \
  --checkpoint checkpoints/microduck_rollers/balanced_model_11600.pt
```

训练和续训示例请参考 [scripts/resume_microduck_rollers.py](scripts/resume_microduck_rollers.py) 以及 [完整使用说明](MICRODUCK_ROLLERS.zh-CN.md)。

## 仓库结构

```text
motrix_envs/   仿真环境与奖励设计
motrix_rl/     PPO/RSLRL 任务接入
scripts/       评估与续训入口
test/          环境、模型和奖励专项测试
checkpoints/   带说明的示例模型
docs/          设计方案与实验记录
```

## 当前状态

- ✅ 被动轮场景与交替蹬滑奖励周期已实现
- ✅ 已提供示例模型、确定性评估和回归测试
- 🚧 长时轮滑稳定性与训练质量仍在持续迭代

## 与 MotrixLab 的关系

本仓库是 [MotrixLab](https://github.com/Motphys/MotrixLab) 的实验扩展，底层使用 [MotrixSim](https://github.com/Motphys/motrixsim-docs)。代码按便于审阅、回移上游和复现实验的方式组织。

## 许可

Apache-2.0。详细信息见上游项目的 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
