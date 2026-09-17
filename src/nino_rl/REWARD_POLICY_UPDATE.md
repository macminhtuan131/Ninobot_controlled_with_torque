> Historical revision. For the current reward, CUDA setup, metrics and training contract,
> use [RL_IMPROVEMENTS.md](RL_IMPROVEMENTS.md) and the [root README](../../README.md).

# Nino reward and history-policy update

Patch base: `add_rl` at `0cb9de53c5b3e2f447a27a36087febd57209b168`.
The active implementation remains `nino_rl/control_v2.py` and YAML `reward_v2`.
Training metadata revision is 3; observation and action semantics remain v2.

## 1. What the repository actually controls

The URDF instantiates two driven wheels **and two front casters**. This is a
supported differential-drive AMR. It is not the free-balancing two-contact TWIP
described in the supplied literature summary. Its learned task is Nav2-guided
path following over cables, with a tradeoff between progress, disturbance,
traction and effort. There is no inverted-pendulum balancing controller here.

Retain the useful ideas: normalized tracking errors, short observation history,
separate actor/critic, effort and rate regularization, and bounded tracking
kernels. Do not insert a bicycle steering-torque action or an unverified TWIP
slope-equilibrium formula into this model. Static pitch on a supported robot
depends on geometry and contacts. The existing attitude deadbands allow small
terrain-induced tilts without requiring exactly zero pitch.

The numbered references in the supplied summary were not accompanied by a
bibliography. In particular, the claimed 71% improvement is not evidence about
this robot. No NMPC or formal safety guarantee is added. Bounded rewards do not
guarantee bounded PPO gradients; KL stopping and gradient clipping are separate
optimizer protections.

## 2. Reward

Notation: h = dt / 0.1 s; C(z) = min(z², 9);
K(e, sigma) = 1 - exp(-min((e/sigma)², 81)).
The kernel is nonnegative and at most 1. All weights are **initial engineering
choices requiring evaluation**, not experimentally optimized values.

The total step reward is the sum of the following rows. Units are metres,
seconds, radians and Nm.

| Term | Implemented formula |
|---|---|
| Progress | 20 (remaining_previous - remaining_current) |
| Cross-track | -0.8 h C(e_y / 0.30) |
| Heading | -0.2 h C(e_psi / 0.35) |
| Linear reference tracking | -h w_v K(v_ground - v_ref, 0.20) |
| Yaw-rate reference tracking | -0.15 h K(omega_ground - omega_ref, 0.50) |
| Vertical impact | -0.05 impact_scale integral(min((a_world_z / 2)⁴, 81) dt) / 0.1 |
| Body rates | -0.05 h [C(gyro_x) + C(gyro_y)] |
| Attitude | -0.5 h [C(max(abs(roll)-0.20,0)/0.15) + C(max(abs(pitch)-0.30,0)/0.15)] |
| Slip | -0.1 h [C(slip_left/0.30) + C(slip_right/0.30)] |
| Action change | -0.02 sum((a-a_previous)²) / h |
| Total effort | -0.04 h mean((tau_applied/2.5)²) |
| Residual effort | -0.005 h mean((tau_residual/0.5)²) |
| Total torque change | -0.02 mean(((tau_applied-tau_previous)/2.5)²) / h |
| Goal braking | -0.15 h exp(-(endpoint_distance/0.80)²) [C(v_ground/0.20) + C(omega_ground/0.30)] |
| Time | -0.01 h |
| Stall | -0.5 h if the existing 3-second stall detector activates |
| Terminal | +100 success; -100 collision/rollover/wrong direction; -75 off path; -50 deadline |

Here w_v = 0.40 for overspeed in the commanded direction or any motion under a
zero linear command, otherwise 0.10. Specifically, overspeed is detected by
(v_ground-v_ref)*v_ref > 0, with abs(v_ref) < 1e-6 treated as a zero command.
Negative references are handled symmetrically. `impact_scale` retains the
existing curriculum ramp min(1, 0.25 + (phase-1)/5).

### Why these changes

- **Reference tracking:** capture the unscaled Nav2 command before applying the
  action. Scaling the reward's reference by the policy's own speed action would
  let a stop command erase its own tracking error. Ground-truth speeds are used
  only in simulated rewards, so spinning wheels cannot satisfy this term.
  Invalid Nav2 references contribute zero tracking terms; the existing stale
  navigation abort remains active. `/cmd_vel_nav` is before the collision
  monitor; actuator stop authority still takes precedence over this soft cost.
- **Terrain slowdown:** moderate underspeed costs less than overspeed. Impact
  and slip penalties can therefore justify slowing for a cable. The target is
  not inferred from private cable-spawn coordinates. Terrain adaptation remains
  reactive unless a valid sensor-based preview producer is supplied.
- **No standing reward:** the exponential terms are negative costs. They add no
  positive per-tick survival reward for staying still. Completion/progress still
  provide the main incentive; this is not a proof against every reward exploit.
- **Real controller effort:** `/wheel_torque_applied` reports the post-limit
  PI-plus-residual effort command. Penalizing only the residual ignored most
  possible actuator effort. The 2.5 Nm normalization matches this repository's
  2.0 Nm PI limit plus 0.5 Nm residual limit. It is a normalization, not a new
  actuator limit. Update it if these operating limits change.
- **Effort is not energy:** torque-squared is an effort/heating proxy. Actual
  electrical energy requires motor current/voltage/efficiency information. The
  torque-rate term uses consecutive policy-step feedback samples and does not
  measure every oscillation of the 1 kHz controller. IMU impacts still use the
  existing full timestamped 50 Hz sample window.
- **Progress:** remove per-step clipping, which could reward two small forward
  steps followed by a larger clipped backward step. Signed differences telescope
  on a fixed path for an undiscounted closed loop. This is not discount-corrected
  potential shaping, and no policy-invariance theorem is claimed. The existing
  frozen-path and common-frame reprojection logic remains necessary.
- **Goal braking:** localize the motion penalty near the endpoint. It gives
  dense guidance for the existing stopped-arrival success test.
- **Mission deadline:** the configured 120-second task budget now returns
  `terminated=True, truncated=False`, preventing SB3's time-limit bootstrap.
  An unrelated external rollout cutoff would require truncation instead.
  Failure takes precedence over success, which takes precedence over deadline.

All time-density terms use simulation dt. Dividing squared action/torque changes
by h approximates an integrated rate cost for a linear ramp. This does not make
the entire learning problem invariant to control frequency: history duration,
discounting and sampling also change.

## 3. Policy and observations

Keep the existing ordered 5 x 60 observation stack. It contains local path
lookahead, heading sin/cos, body/wheel velocities, IMU, lidar sectors, Nav2
references, previous action and optional valid terrain preview. No simulator
truth velocity or slip is added to the actor. Five frames span about 0.4 seconds
between oldest and newest samples at 10 Hz; this is finite memory, not an exact
integral state or a proof of the Markov property.

For each actor and critic independently, form

    z_t = concat(o_t, o_t - o_(t-1), f_history(o_(t-4), ..., o_t)).

`f_history` is Conv1d(60 -> 32, k=3, padding=1), ELU,
Conv1d(32 -> 32, k=3, padding=1), ELU, flatten, Linear(160 -> 64), ELU.
It sees only past/current frames. The newest state has a direct bypass, so
history compression cannot discard the current command or IMU state. Actor and
critic have separate trainable extractors to avoid forcing the same features
to optimize both objectives. There is no recurrent hidden state to manage.

The actor MLP is 184 -> 128 -> 128 -> 3; the critic is 184 -> 256 -> 128 -> 1.
Both use ELU. SB3 retains its diagonal Gaussian distribution and correct rollout
log-probabilities; it clips sampled actions to [-1,1] for the environment.
No manual tanh transformation is inserted into PPO's probability calculation.

Initial standard deviation is 0.25 instead of SB3's default 1.0. The speed head
has bias 0.4, corresponding to a 0.70 reference scale; the two residual biases
are zero. Small random head weights make the initial mean close to those values,
not exactly identical for every observation. This reduces initial random torque
and stop commands without removing exploration. Standard deviation is learned;
inspect action saturation and entropy during real training.

Action mapping remains

    scale = (a_speed + 1)/2
    delta_tau_L = 0.5 clip(a_forward - a_yaw, -1, 1)
    delta_tau_R = 0.5 clip(a_forward + a_yaw, -1, 1).

The PI controller, torque/speed/slew limits, atomic command topic and watchdog
remain as implemented in the base revision. In particular, a true zero stop
blocks residual torque; the in-place-turn scale floor still applies. This patch
does not turn the 10 Hz high-level policy into a TWIP balancing loop.

PPO changes: learning rate 3e-4, gamma 0.997 (about a 33-second geometric horizon
at 10 Hz), target_kl 0.015. Retain GAE lambda 0.95, clip 0.2, gradient norm 0.5,
2048-step rollouts, batch size 256 and 10 epochs. KL stopping limits excessively
large updates; it is not a stability guarantee. Start a **new training run**.

## 4. Apply and train

The patch applies to the exact base revision above. It does not modify a remote
branch. Keep local edits; if `git apply --check` reports a conflict, do not force
the patch or reset your work. Reconcile those changes first.

From your repository root, with the downloaded patch in ~/Downloads:

```bash
git status --short
git rev-parse HEAD
git apply --check ~/Downloads/nino_reward_policy_update.patch
git apply ~/Downloads/nino_reward_policy_update.patch
```

Stop any old training/simulator processes, then rebuild in the existing project
venv. The dependency requirements are unchanged.

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
python -m colcon build --symlink-install --packages-up-to nino_rl
source install/setup.bash
ros2 launch nino_rl training_sim.launch.py headless:=true rviz:=false
```

In a second terminal at the repository root, source the same ROS, venv and
workspace setup, then run a fresh phase-1 smoke run:

```bash
ros2 run nino_rl train --phase 1 --timesteps 4096 --checkpoint-every 2048 --check-env
```

If this succeeds, start a longer run:

```bash
ros2 run nino_rl train --phase 1 --timesteps 100000 --checkpoint-every 10000
```

Do **not** pass a pre-patch model to `--resume`. New models store reward/PPO,
history, control frequency, deadline and torque-scale settings in the checkpoint;
resume rejects differences rather than silently keeping old PPO parameters.
Old v2 models still support inference with the unchanged sensor/action interface,
but their returns under this new reward are not comparable to their old returns.

For a checkpoint trained with this patch, resume with its saved configuration:

```bash
ros2 run nino_rl train --config /absolute/run/ppo.yaml --resume /absolute/run/nino_ppo_final.zip --phase 2 --timesteps 100000
```

Phase changes are allowed; advance only after held-out evaluation. `--timesteps`
means additional steps on resume. The custom policy module must stay installed
in the sourced workspace for evaluation/deployment to load these checkpoints.

## 5. Evaluate the hypothesis

Run controllers sequentially against the same simulator, phases and held-out
seeds. Use the model's saved configuration in both evaluations:

```bash
ros2 run nino_rl evaluate --config /absolute/run/ppo.yaml --model /absolute/run/nino_ppo_final.zip --phase 1 --episodes 30 --seed 10000 --output rl_runs/eval_new_phase1
ros2 run nino_rl evaluate_baseline --config /absolute/run/ppo.yaml --phase 1 --episodes 30 --seed 10000 --output rl_runs/eval_baseline_phase1
```

Repeat after each terrain phase; pass `--randomized` to both commands for a
matched randomized comparison. Compare old v2 policy inference on the same
episodes as a third controller if you have a checkpoint. For training-curve
claims, use multiple training seeds and keep hyperparameter-selection seeds
separate from final test seeds.

Prioritize success rate, then successful completion time, RMS/max path error,
peak/RMS vertical acceleration, slip and RMS total wheel torque. Never select
the smoothest policy if it achieves that by failing to finish. Report failures
alongside successful-run metrics, and compare physical metrics rather than raw
return across reward versions. A suggested progression gate is at least 90%
held-out success plus acceptable tracking/impact metrics; 30 episodes provide
an initial screening result, not a precise performance guarantee.

The optional preview still needs a real producer. Existing domain randomization
does not implement a full contact-friction/mass model. Torque effort/rate are
sampled at policy boundaries; a future high-bandwidth actuator integral would
be needed for a full motor-heating or chatter objective. No terrain performance
gain is claimed until the above experiments have run.

## 6. Local validation

Result on the delivered patch: **63 portable tests passed**, Python syntax
compilation passed, and whitespace/error checks passed. Tested with NumPy
1.26.4, PyTorch 2.14.0+cpu, Stable-Baselines3 2.9.0 and Gymnasium 1.3.0.
The initial full-suite collection stopped at the existing ROS-only test because
this environment has no `rclpy`; that test was not claimed as passing.

Portable tests, including real PyTorch/SB3 optimization and serialization:

```bash
PYTHONPATH=src/nino_rl:src/nino_control python -m pytest src/nino_rl/test src/nino_control/test --ignore=src/nino_rl/test/test_navigation_frames.py -q
```

The excluded existing test imports ROS transport and needs `rclpy`. Run the full
suite in a sourced ROS workspace. The included environment/actuator tests execute
production methods with fake transport; they are not Gazebo integration tests.
The policy test uses a synthetic Gymnasium environment to verify API behavior,
optimization, gradients, save/load and continued training. It does not simulate
robot dynamics. No ROS/Gazebo training was run in the patch-building environment.

## 7. Primary references

- [PPO paper](https://arxiv.org/abs/1707.06347): clipped policy optimization.
- [SB3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html):
  frame stacking, target KL, gradient clipping and distribution parameters.
- [SB3 custom policy documentation](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html):
  custom extractors, separate actor/critic extractors and action clipping.
- [Gymnasium time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/):
  task termination versus external truncation and value bootstrapping.

The weights, history architecture, asymmetric tracking penalty and effort terms
above are proposed for this repository; these sources do not establish their
optimality or a quantitative improvement on Nino.
