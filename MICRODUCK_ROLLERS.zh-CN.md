# MicroDuck 被动轮轮滑（纯仿真）

这是 MotrixLab 的 MicroDuck 被动轮轮滑实验分支。目标是先训练稳定、丝滑、接近人类节奏的直线轮滑，再以这套基础策略继续训练转弯和花式动作。

> 当前阶段只用于电脑仿真，不用于真实机器人。精选模型是当前长训结果中的最佳候选，并不代表“冠军级动作”已经最终完成。

## 已实现内容

- 四个轮子均为无电机驱动的被动轮，前进动力来自腿部动作；
- 策略仍只控制 14 个舵机关节，观察维度保持 61；
- 动作滤波和非对称响应，减少快速碎步和关节抖动；
- 双腿蹬地侧识别、左右交替奖励和弱侧补偿；
- 每条腿只有完成“蹬地 → 离地 → 重新接触”才计为完整周期；
- 支撑脚切换、直线航向、侧向漂移和滑行效率约束；
- 支持高频检查点、从指定模型续训以及固定随机种子评估；
- 评估会输出稳定率、距离、平滑度、左右蹬地比例、完整周期和偏航指标。

环境名称：

```text
microduck-flat-tile-passive-rollers
```

## 目录说明

```text
checkpoints/microduck_rollers/       精选模型与模型说明
motrix_envs/.../microduck/            环境、奖励和轮滑模型
motrix_rl/.../tasks/microduck.py      PPO 训练配置
scripts/eval_microduck_rollers.py     批量评估
scripts/resume_microduck_rollers.py   从检查点继续训练
test/test_microduck_rollers_*.py      环境、物理模型和奖励测试
```

大体积的训练日志、缓存和其余数万个检查点没有上传。它们对运行精选模型不是必需的。

## 1. 安装

建议使用带 NVIDIA 显卡的 Windows 或 Linux 电脑。先安装 [Git LFS](https://git-lfs.com/) 和 [uv](https://docs.astral.sh/uv/)，然后在仓库根目录执行：

```powershell
git lfs pull
uv sync --all-packages --extra rslrl
```

## 2. 直接观看精选模型

```powershell
uv run python scripts/play.py `
  --env microduck-flat-tile-passive-rollers `
  --sim-backend np `
  --policy checkpoints/microduck_rollers/balanced_model_11600.pt `
  --rllib rslrl `
  --num-envs 1 `
  --seed 1234
```

关闭渲染窗口即可停止。

## 3. 复核量化结果

单环境复现视觉评估：

```powershell
uv run python scripts/eval_microduck_rollers.py `
  checkpoints/microduck_rollers/balanced_model_11600.pt `
  --num-envs 1 `
  --seed 1234
```

更可靠的批量评估：

```powershell
uv run python scripts/eval_microduck_rollers.py `
  checkpoints/microduck_rollers/balanced_model_11600.pt `
  --num-envs 100 `
  --seed 1234
```

重点关注：

- `success_full_episode`：完整 20 秒内没有摔倒的比例，越高越好；
- `mean_push_side_balance`：左右蹬地均衡度，1 最好；
- `mean_*_complete_fraction`：每条腿从蹬地到离地再落地的完成比例；
- `mean_action_smoothness_cost`：动作变化代价，越低越好；
- `mean_abs_yaw_rate_rad_s` 与 `mean_heading_change_per_meter_rad_m`：转圈倾向，越低越好。

## 4. 从精选模型继续训练

先进行短训验证：

```powershell
uv run python scripts/resume_microduck_rollers.py `
  checkpoints/microduck_rollers/balanced_model_11600.pt `
  --iterations 200 `
  --num-envs 256 `
  --learning-rate 0.00001 `
  --seed 1234
```

验证稳定后再增加 `--iterations`。训练产物会写入 `runs/`，该目录已被 Git 忽略。

## 5. 从零训练

```powershell
uv run python scripts/train.py `
  --env microduck-flat-tile-passive-rollers `
  --sim-backend np `
  --rllib rslrl `
  --num-envs 256 `
  --seed 1234
```

从零训练耗时较长，而且最终轮次不一定是视觉效果最好的轮次。建议定期使用评估脚本筛选检查点，并对候选模型逐个渲染。

## 当前精选结果

`balanced_model_11600.pt` 是对六个保留检查点使用同一套固定种子评估后筛出的当前最佳综合基线。固定随机种子 `1234`、单环境、20 秒结果：

| 指标 | 结果 |
|---|---:|
| 20 秒稳定率 | 100% |
| 前进距离 | 3.206 m |
| 左 / 右蹬地比例 | 50.9% / 49.1% |
| 左右均衡度 | 0.982 |
| 蹬地侧切换率 | 0.679 |
| 左 / 右完整周期率 | 97.1% / 90.9% |
| 动作平滑代价 | 0.070796 |
| 综合平滑分 | 0.218 |
| 平均绝对偏航速度 | 0.3403 rad/s |
| 20 秒累计偏航 | 181.61° |
| 每米航向变化 | 0.9888 rad/m |

这些数据说明双脚使用和完整蹬滑周期已经较均衡，动作也比长训末段检查点更平滑。不过累计偏航仍约半圈，尚未达到基本直线，更不能称为冠军级最终动作。后续优化应首先显著压低偏航，再继续提高右腿完整周期率，并避免为了单一奖励牺牲自然动作。

## 测试

```powershell
uv run pytest
```

核心测试覆盖被动轮无驱动约束、61/14 观察动作接口、奖励有限性、左右交替、完整蹬滑周期以及回放时课程阶段。
