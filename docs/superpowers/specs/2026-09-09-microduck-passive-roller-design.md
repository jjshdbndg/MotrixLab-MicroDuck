# MicroDuck Passive-Roller Locomotion Design

Date: 2026-09-09

## Objective

Add a separate MotrixSim/MotrixLab task that trains a physical-size MicroDuck with the official passive roller kit to propel itself by leg push-off and balance control on smooth tile. The first milestone is straight-line locomotion at 0.2 m/s for at least 1 m while remaining upright for 10 seconds. Wheels receive no actuator commands.

This milestone is simulation-only. Deployment to a physical robot is explicitly out of scope until actuator, contact, and safety validation is complete.

## Source of truth

Use the official `pollen-robotics/microduck_rl` develop branch as the reference for:

- `robot_groundcontact_rollers.xml` geometry, mass, inertia, wheel placement, and passive joint definitions;
- the `Mjlab-Velocity-Flat-MicroDuck-Rollers` task structure;
- the shared 61-dimensional actor observation and 14-dimensional action contracts;
- XL330/BAM actuator behavior, command delay, friction, battery-voltage variation, and existing domain-randomization ranges;
- roller-task reward and termination conventions.

The official repository is kept as a read-only sibling checkout. Selected behavior is ported rather than vendoring the entire training framework. Original notices and the non-commercial share-alike asset license remain attached to copied model assets.

## Approach

Create a new roller environment alongside the existing walking environment. Do not modify the semantics or registration of `microduck-flat-terrain-walk`.

The new environment is registered as `microduck-flat-tile-passive-rollers`. It uses MotrixSim's NumPy physics backend and RSL-RL/PyTorch PPO. The environment preserves the current policy interface so later policies can be exported and compared without observation or action adapters.

Alternatives rejected:

- Adding approximate cylinder wheels directly to the walking model is faster but loses authoritative geometry and Sim-to-Real value.
- Training only in the official mjlab stack gives a useful reference policy but does not establish the requested MotrixSim environment.

## Components

### Reference checkout

Clone `pollen-robotics/microduck_rl` beside the MotrixLab project on drive D and record the inspected commit in the port documentation. Treat it as read-only input.

### Robot and scene models

Add roller-specific MJCF files under the MicroDuck environment assets. Import the official roller assembly's dimensions, transforms, inertial properties, and passive wheel hinges. Servo joints remain the existing 14 controlled joints. Every wheel joint is named with a `passive_` prefix and is excluded from actuators, policy actions, pose rewards, and controlled-joint observations.

The tile scene is flat and rigid. Initial wheel/ground contact and rolling-resistance parameters come from the official roller task. MotrixSim parameters are mapped explicitly and documented; unsupported parameters are approximated with the closest joint damping, joint friction-loss, and contact-friction terms rather than silently dropped.

### Environment

Add a dedicated configuration and environment implementation. Preserve the 61-value actor observation layout:

- 48 proprioceptive and policy-history values compatible with the existing MicroDuck policy contract;
- 13 command values: twist (3), head pose (4), and body pose (6).

The action remains 14 servo position targets. Passive wheel position and velocity may be read internally for rewards and diagnostics but do not change the actor observation contract in this milestone.

### Training configuration

Add an RSL-RL PPO task configuration derived from the existing MicroDuck task and the official roller task. Start with 256 parallel environments for smoke and stability runs, then use 512 when memory and throughput checks pass on the RTX 3060 Laptop GPU. Checkpoints are saved every 50 PPO iterations. The training entry point must support resuming from a selected checkpoint before any long run begins.

## Curriculum and rewards

Training progresses automatically through three stages using environment control steps:

1. **Roller balance:** hold the nominal posture on passive wheels, minimize lateral drift, and survive without wheel actuation.
2. **Push and glide:** introduce forward commands up to 0.2 m/s and reward alternating leg push-off followed by efficient coasting.
3. **Straight velocity tracking:** sample the milestone command, penalize heading error and lateral displacement, and require sustained travel.

Reward terms cover:

- forward velocity and distance progress;
- upright posture, root height, and heading stability;
- low lateral velocity and lateral displacement;
- wheel rolling consistency and forward roll direction;
- alternating support/push timing;
- glide efficiency after push-off;
- smooth actions, bounded joint velocity/acceleration, and joint-limit avoidance;
- termination on falls or excessive tilt.

The curriculum does not include turning, reverse skating, crouching, slopes, spins, or trick motions.

## Smooth-tile parameterization

The nominal model uses official roller parameters. Domain randomization is deliberately narrower than walking-task randomization during the first milestone:

- small variation in wheel radius and assembly mass to cover manufacturing tolerances;
- small variation in passive-joint damping/friction loss to cover bearing differences;
- bounded wheel/tile sliding friction variation around the official nominal value;
- existing actuator and command-delay variation from the official Sim-to-Real recipe.

Before physical deployment, nominal rolling resistance is calibrated from a coast-down test on the intended smooth tile, and lateral contact friction is calibrated from a low-speed sideways pull test. Those measurements refine the distributions without changing the environment interface.

## Data flow

At each control step, the policy receives the shared observation and outputs 14 target offsets. MotrixSim applies the servo targets, integrates body and passive-wheel dynamics, and exposes body, foot, contact, and wheel state. The environment computes rewards, termination, curriculum state, and diagnostic metrics. RSL-RL batches transitions for PPO and writes checkpoints and TensorBoard summaries under a roller-specific run directory.

## Diagnostics

TensorBoard and rollout logs include:

- mean reward and episode length;
- commanded and measured forward velocity;
- forward distance and lateral drift;
- root roll/pitch and fall rate;
- per-wheel angular velocity and rolling-consistency error;
- alternating-push score and glide fraction;
- action-rate and joint-limit penalties;
- curriculum stage distribution.

Visual playback is required to reject reward-hacking behaviors such as hopping, foot dragging, wheel skidding, or pitching the body to fall forward.

## Error handling

- Model loading fails with a named diagnostic when expected wheel bodies, passive joints, sites, or actuators are missing.
- Reward computation guards against NaN/Inf and reports the offending term.
- Unsupported MotrixSim contact parameters are recorded in the port mapping and covered by behavioral tests.
- A long training run is not started unless a short headless rollout and PPO smoke run complete successfully.
- Existing walking-task behavior must remain unchanged.

## Testing

1. **Model invariants:** two passive roller assemblies exist, wheel joints have no actuators, 14 servo actuators remain, mass and transforms match the recorded official source.
2. **Interface invariants:** actor observation shape is `(num_envs, 61)` and action shape is `(num_envs, 14)`.
3. **Physics behavior:** wheels rotate freely around the intended axes, no drive torque is applied, forward push produces forward roll, lateral force produces substantially more resistance than forward rolling.
4. **Reward tests:** correct sign, finite output, and expected relative ordering for upright glide, lateral skid, fall, hopping, and alternating push fixtures.
5. **Regression:** current MicroDuck walking smoke test still passes.
6. **Training smoke:** a small run completes updates, writes a checkpoint, resumes from it, and produces finite metrics.
7. **Milestone evaluation:** across deterministic evaluation episodes, the policy averages at least 0.2 m/s command tracking, travels at least 1 m, and remains upright for 10 seconds; visual review confirms leg-powered push-and-glide behavior.

## Delivery sequence

1. Capture official reference commit and parameter mapping.
2. Port and validate the passive-roller MJCF model.
3. Add environment registration and physics diagnostics.
4. Add reward terms, curriculum, and tests.
5. Add resumable RSL-RL configuration and smoke training.
6. Tune on 256 environments, scale to 512, and evaluate the milestone policy.

Further roller tricks are a later design cycle based on the trained straight-skating policy.
