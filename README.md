# Quadruped Robot Control Using Reinforcement Learning and Guidance Loop

PPO-based reinforcement learning controller and guidance loop for the **Unitree Go2** quadruped robot, with position tracking and disturbance rejection in Isaac Gym.

> **Thesis:** B.Sc. Project  
> **Department:** Aerospace Engineering, Sharif University of Technology  
> **Author:** Sepehr Mahfar  
> **Supervisor:** Dr. Alireza Sharifi  
> **Date:** September 2025

---

## Overview

This project develops a complete locomotion and navigation stack for the Unitree Go2 quadruped robot. The control architecture consists of two layers: a PPO-trained deep neural network that converts velocity commands into desired joint positions, and a PD controller that computes the torques required to reach those positions. On top of this, a guidance loop translates position commands into velocity commands, enabling the robot to autonomously navigate to a target location.

Two guidance strategies are implemented and compared. In the first, the robot aligns its heading toward the target and moves along its body x-axis. In the second, velocity commands are issued simultaneously along both body x and y axes, enabling more direct lateral motion.

All training and evaluation is performed in NVIDIA Isaac Gym using the `legged_gym` environment and the `rsl_rl` library, with several modifications to the reward function, hyperparameters, and guidance modules.

---

## Methods

### Controller 1 — PPO Policy

The first controller is a deep neural network trained via Proximal Policy Optimization (PPO) using an actor-critic architecture. It takes robot observations as input and outputs desired joint positions for all 12 degrees of freedom. The policy is trained to track linear and angular velocity commands while keeping the robot balanced.

**Network architecture:** 3 hidden layers of 512 → 256 → 128 nodes with ELU activation, shared structure for both actor and critic.

**Observations include:** linear and angular velocities of the base, joint positions and velocities, roll and pitch angles, control commands, and previous actions.

**Action space:** Continuous Gaussian distribution over desired joint positions, scaled and offset from the default joint configuration:
```
Desired DOF Positions = (Action Scale × Action) + Default Angle
```
where `Action Scale = 0.25`.

**PPO hyperparameters:**

| Parameter | Value |
|---|---|
| Number of environments | 4096 |
| Clip parameter ε | 0.2 |
| Value function loss coefficient | 1.0 |
| Entropy coefficient | 0.01 |
| Epochs per update | 5 |
| Mini-batches | 4 |
| Initial learning rate | 0.001 |
| Discount factor γ | 0.99 |
| GAE smoothing λ | 0.95 |

**Reward function:** The total reward is a weighted sum of the following terms:

| Term | Scale |
|---|---|
| Linear velocity tracking (x, y) | +1.0 |
| Angular velocity tracking (z) | +0.5 |
| Linear velocity penalty (z body) | −2.0 |
| Angular velocity penalty (x, y body) | −0.05 |
| Joint torque penalty | −0.0002 |
| Joint acceleration penalty | −2.5×10⁻⁷ |
| Foot air time reward | +1.0 |
| Collision penalty (shank/thigh) | −1.0 |
| Action rate penalty | −0.01 |
| Joint limit proximity penalty | −10.0 |

**Domain randomization** is applied to improve sim-to-real transfer: ground friction coefficient is randomized in [0.5, 1.25], proportional observation noise is added to all measurements, and random velocity pushes of up to 1 m/s are applied to the robot's CoM every 15 seconds.

### Controller 2 — PD Torque Controller

A proportional-derivative controller converts the desired joint positions output by the policy into joint torques:

```
τ = kp × (q_desired − q) − kd × q̇
```

where `kp = 20 N·m/rad` and `kd = 0.5 N·m·s/rad`, applied uniformly across all 12 joints.

### Guidance Loop

The guidance loop converts position commands into velocity commands that the PPO policy can track.

**Approach 1 — Heading + Forward motion:**
A PID controller regulates the robot's forward speed based on the Euclidean distance to the target, while a proportional heading controller aligns the robot's yaw with the direction to the goal before moving forward.

**Approach 2 — Direct x/y velocity:**
Two independent PID controllers issue simultaneous velocity commands along the body x and y axes, enabling faster lateral motion without an explicit heading alignment step.

---

## Requirements

- Python 3.8
- [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym)
- [`legged_gym`](https://github.com/leggedrobotics/legged_gym)
- [`rsl_rl`](https://github.com/leggedrobotics/rsl_rl)
- PyTorch ≥ 1.10
- NVIDIA GPU with CUDA support

---

## Installation

**1. Install Isaac Gym**

Download Isaac Gym Preview 4 from the [NVIDIA developer site](https://developer.nvidia.com/isaac-gym) and follow the installation instructions in the package:
```bash
cd isaacgym/python
pip install -e .
```

**2. Install `rsl_rl`**

```bash
git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl
pip install -e .
```

**3. Install `legged_gym`**

```bash
git clone https://github.com/leggedrobotics/legged_gym.git
cd legged_gym
pip install -e .
```

**4. Clone this repository and apply modifications**

```bash
git clone <this-repo-url>
cd <this-repo>
pip install -e .
```

This repository contains modifications to the base `legged_gym` environment including a custom reward function, PPO hyperparameters, and the guidance loop module. The modified files override the corresponding base files — do not copy the base `legged_gym` files on top of this repo's versions.

---

## Usage

### Training

To train the PPO policy from scratch on the Unitree Go2:

```bash
cd legged_gym/scripts
python train.py --task=go2 --headless
```

To monitor training progress with Isaac Gym's viewer:

```bash
python train.py --task=go2
```

Training logs and model checkpoints are saved to `logs/go2/<run_name>/`.

### Playing a Trained Policy

To load a checkpoint and run the policy in simulation:

```bash
python play.py --task=go2
```

By default this loads the latest checkpoint. To specify a run:

```bash
python play.py --task=go2 --load_run=<run_name> --checkpoint=<iteration>
```

### Running the Guidance Loop

To enable position tracking using **Approach 1** (heading + forward motion):

```bash
python play.py --task=go2 --guidance=approach1 --target_x=20.0 --target_y=10.0
```

To use **Approach 2** (direct x/y velocity):

```bash
python play.py --task=go2 --guidance=approach2 --target_x=20.0 --target_y=10.0
```

### Sinusoidal Path Tracking

To evaluate trajectory tracking on a sinusoidal path:

```bash
python play.py --task=go2 --guidance=approach1 --path=sine
```

The reference trajectory follows `x_desired = 0.5t`, `y_desired = sin(0.5t)`.

---

## Results

The trained policy successfully navigates to target positions in both ideal and disturbed conditions. Key findings from simulation:

- The robot reaches a target at (20 m, 10 m) from a random starting position within a reasonable time with good accuracy under both ideal conditions and external disturbances.
- Random velocity pushes of 1 m/s cause transient deviations but the robot recovers and continues toward the target.
- Sinusoidal trajectory tracking is achieved with high fidelity even in the presence of observation noise and friction randomization, demonstrating the robustness of the learned policy.

---

## Citation

If you use this work, please cite:

```
S. Mahfar, "Quadruped Robot Control Using Reinforcement Learning and Guidance Loop,"
B.Sc. Thesis, Dept. of Aerospace Engineering, Sharif University of Technology, Sep. 2024.
Supervisor: Dr. A. Sharifi.
```
