# RL engineering update: reward, policy and evaluation

Base: `add_rl` commit `d015b8239648c64974e5badd041aa87f3a5a0fdc`.
Training contract revision: **24**. No trained model is bundled.
The [root README](../../README.md) is the authoritative installation-to-training guide.

## Decisions from the supplied research report

| Proposed idea | Decision for this repository | Reason |
|---|---|---|
| Residual RL over a classical controller | Retain Nav2 + wheel PI + PPO | Already implemented; preserves useful baseline control and actuator limits |
| History-dependent policy | Retain independent actor/critic history encoders | Existing five-frame history can represent recent shocks/delay without recurrent state management |
| Pitch/ZMP reference for inverted pendulum | Do not apply | This URDF has two passive caster supports; its pitch is not the report's unsupported pendulum state. The report also needs careful slope/gravity coordinate definitions |
| SWAE elevation-map encoder | Defer | The provided LiDAR is 2D and no elevation-map dataset/producer is present. A latent vector cannot create missing height measurements |
| ESKF + LiDAR odometry | Defer | Current localization uses AMCL, wheel odometry and TF; a new estimator needs a sensor/observability design and independent validation |
| Asymmetric actor-critic | Defer | Both existing networks receive sensor history. Gazebo truth is used for rewards only; separate network weights do not make a critic privileged |
| Isaac Lab thousands of parallel robots | Separate future backend | ROS/Gazebo services, Nav2, reset and contact dynamics need a real port. CUDA PPO alone does not convert Gazebo physics to GPU parallelism |
| Curriculum/domain randomization | Strengthen existing six-phase curriculum | Add perturbation ramp, correct simulation-time delay and preserve zero-residual baseline |
| Torque saturation penalty | Implement | Discourage spending time at the effective PI + residual operating envelope |
| Automatic trajectory quality | Implement | RMSE alone hides timing/coverage failures; add P95, heading, endpoint, completion and matching baseline evaluation |
| Guaranteed zero-shot transfer / immunity to noise | Reject claim | Neither the attached prose nor software tests establish these outcomes |

Most of the sound RL architecture was already present at the base commit. This
update deliberately fixes execution, measurement and reproducibility before
adding another algorithm. Reported performance percentages in the attachment
are not treated as results for Nino.

## Policy: actual implementation

PPO runs at a nominal **10 Hz**. The underlying effort-drive loop is configured
at **500 Hz**, matching the 2 ms training-world physics step. Controller limits
and watchdogs remain in `nino_control`.

Each frame has 60 values. Five frames, oldest to newest, give a 300-dimensional
observation. Frames include nine robot-relative path lookaheads, heading sine
and cosine, body/wheel velocities, IMU quaternion/gyro/acceleration, five LiDAR
sectors, Nav2 reference/waypoint context, previous actions and optional terrain
preview fields. The current configuration requires valid preview data. The actor
excludes simulator ground-truth velocity and slip.

The terrain preview is active by default and comes from a 20 Hz downward-looking
31-ray LiDAR fan. It reports the nearest non-flat relief plus signed left/right
height, allowing the policy to change speed before a low cable or basin reaches
the caster wheels. Every lockstep policy step requires a fresh preview; the
hazard's hidden spawn coordinates are not passed to the actor.

For the actor and critic separately:

\[
z_t=[o_t,\;o_t-o_{t-1},\;f(o_{t-4:t})]\in\mathbb R^{184}.
\]

`f` is Conv1d(60→32, kernel 3, padding 1), ELU, Conv1d(32→32, kernel 3,
padding 1), ELU, flatten, Linear(160→64), ELU. The actor MLP is
184→128→128→3; the critic is 184→256→128→1, with ELU activations. There is
no recurrent hidden state. This architecture is retained from the base revision.

The actor samples a diagonal Gaussian during training; SB3 uses its original
log probabilities and clips the executed actions to [-1,1]. Deterministic
evaluation uses the mean. This is not a manually squashed/tanh Gaussian.
Initial standard deviation is 0.25; initial speed bias is 0.4 (approximately
70% speed), with zero residual biases. Small initial head weights make the
mean slightly state-dependent. Exploration variance is learned.

For actions \(a=[a_s,a_f,a_y]\):

\[
s=(a_s+1)/2,\quad
\Delta\tau_L=0.5\,\mathrm{clip}(a_f-a_y,-1,1),\quad
\Delta\tau_R=0.5\,\mathrm{clip}(a_f+a_y,-1,1).
\]

The controller scales the Nav2 reference, calculates PI wheel torque, adds
residuals and applies torque/speed/slew/watchdog guards. The active simulation
config limits PI to 4 Nm per wheel and the residual action to 2 Nm per wheel;
the final actuator safety limit and slew limiter still apply. A true zero
speed-scale command blocks residual drive. The existing in-place-turn scale
floor and filtered scaling remain. This is not direct unsupported balancing.

PPO: learning rate 0.0003, gamma 0.997, GAE lambda 0.95, rollout 2048,
batch 256, epochs 10, clip 0.2, entropy coefficient 0.001, value coefficient
0.5, gradient norm limit 0.5, target KL 0.015. These values remain unchanged.
CUDA is now the training default, and `auto` resolves to CUDA with an explicit
availability check. Explicit CPU remains available for software diagnostics.

The clipped update follows [SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).
A KL threshold is an update safeguard, not a guarantee of convergence or robot
stability. Convergence and throughput still need on-machine measurement.

## Reward: exact active defaults

Only `control_v2.compute_reward` and YAML `reward_v2` are active. The legacy
`core.compute_reward` and YAML `reward` remain for historical regression tests.

Let \(h=\Delta t/0.1\), \(C(x)=\min(x^2,9)\),
\(K(e,\sigma)=1-\exp[-\min((e/\sigma)^2,81)]\).
Let \(d\) be remaining reference arc length, \(e_y\) signed perpendicular
path error, \(e_\psi\) wrapped heading error, \(\tau\) the post-limit total
controller effort, and \(\delta\tau\) the perturbed residual command.
The total reward is the sum of these terms:

| Term | Formula/default |
|---|---|
| Signed progress | \(20(d_{t-1}-d_t)\), gated by path alignment for forward motion |
| Lateral | \(-0.50h C(e_y/0.25)\) |
| Heading | \(-0.25h C(e_\psi/0.35)\) |
| Linear reference tracking | \(-w_v h K(v-v_{ref},1.20)\) |
| Yaw reference tracking | \(-0.15h K(\omega-\omega_{ref},0.50)\) |
| Impact | \(-0.05 q\int \min((|a_z^w|/2)^4,81)dt/0.1\) |
| Body rates | \(-0.05h[C(\dot\phi)+C(\dot\theta)]\) |
| Excess tilt | \(-0.5h[C((|\phi|-0.20)_+/0.15)+C((|\theta|-0.30)_+/0.15)]\) |
| Wheel slip | \(-0.1h\sum_{L,R} C(\mathrm{slip}/0.30)\) |
| Action change | \(-0.02\|a_t-a_{t-1}\|^2/h\) |
| Total effort | \(-0.04h\operatorname{mean}((\tau/5.0)^2)\) |
| Residual effort | \(-0.005h\operatorname{mean}((\delta\tau/0.5)^2)\) |
| Total torque change | \(-0.02\operatorname{mean}(((\tau_t-\tau_{t-1})/5.0)^2)/h\) |
| **Operating-envelope saturation** | \(-0.03h\operatorname{mean}[\mathrm{clip}((|\tau|/5.0-0.9)/0.1,0,1)^2]\) |
| Goal braking | \(-0.15h e^{-(d_{end}/0.8)^2}[C(v/1.20)+C(\omega/0.30)]\) |
| Time | \(-0.01h\) |
| Stall | \(-0.5h\) when the existing 3-second progress window flags a stall |
| On-time success | up to +50, proportional to positive margin before the 15-second target |
| Terminal | +100 success; -100 collision/rollover/wrong direction; -75 off path; deadline \(-100[0.5+0.5(1-c)]\) |

Here \(q=\min(1,0.25+(phase-1)/5)\), \((x)_+=\max(0,x)\), and
\(w_v=0.40\) for overspeed in the commanded direction or movement under a
zero command; otherwise \(w_v=0.10\). The reference is captured before the
action and is not scaled by the actor, so stopping does not erase its own
tracking error. Gazebo truth speeds affect reward only. Reward terms are logged
individually in TensorBoard.

The saturation term is zero below 4.5 Nm and reaches -0.03 per 0.1-second
step when both wheels reach 5.0 Nm. This is the effective PI-plus-residual
operating envelope, not the URDF's 12 Nm hard joint limit. Torque squared is an
effort/heating proxy, not measured electrical energy. These are initial weights,
not empirically optimized weights; use matched evaluations to tune them.

The signed progress term avoids an undiscounted forward/backward reward loop.
It is not discount-correct potential shaping and carries no policy-invariance
claim. Path lateral error now uses the true perpendicular component; separate
endpoint/path-distance metrics measure longitudinal overshoot. Reprojection
of the fixed episode path retains current AMCL transforms.

A successful arrival must be inside the 0.10 m endpoint circle with
|heading| ≤12°. Rollover threshold is 35°. The 20-second mission deadline
is a task terminal (`terminated=True`), not an external time-limit truncation.
Failure overrides success, which overrides timeout. Sensor/transport failures
abort the run rather than becoming learned collision penalties.

## Timing and robustness changes

- The IMU subscription buffers 100 samples instead of one. Each policy action
  advances Gazebo by exactly 100 one-ms physics steps while the world otherwise
  remains paused. Impact scoring requires at least 80% interval coverage and a
  bounded endpoint lag; a post-step barrier also requires fresh odometry,
  ground-truth velocity, joints, laser, and applied-torque feedback. Real missing
  coverage still aborts. IMU periods/gaps retain the 0.1 s hold cap.
- Residual command delay is expressed in physics steps, not wall-clock sleep.
  Bounded wall timeouts catch failed Gazebo steps or missing ROS feedback.
  Physics advancement is lockstep; ROS transport and Nav2 remain asynchronous
  and are synchronized at the post-step barrier.
- Domain randomization is disabled by default so cable phase difficulty depends
  only on diameter and angle. When explicitly enabled, its optional strength
  decreases with the requested hard-to-easy phase order.
  `traction_scale`, `motor_delay_ms` and `torque_noise_std_nm` are historical
  config names for **residual-channel gain, command delay and additive noise**.
  They do not randomize SDF friction, mass, CoM or the whole motor plant.
  This limitation is explicit; claiming broad physical domain randomization
  would be misleading. Configured IMU sensor noise in Gazebo remains active.
- Randomized evaluation uses full strength. Baseline always has zero residual
  torque. Both evaluators consume the same random draws and use the same
  metrics pipeline, with no competing simultaneous process.
- Training contract revision 24 includes task, observation and randomization
  settings, while permitting phase changes, seed/device changes and terrain
  curriculum adjustments. Incompatible resumes fail instead of silently
  changing the learning objective. Run metadata records software and GPU.

## Metric definitions and limits

For distances \(e_i\), the time-weighted path RMSE is
\(\sqrt{\sum_i w_i e_i^2}\), with trapezoidal timestamp weights summing to one.
P95 uses the corresponding weighted empirical distance distribution. Path
error is Euclidean distance to the nearest point on a finite polyline;
cross-track error is its signed perpendicular component. At an endpoint these
can differ. Heading RMSE wraps angle errors into [-π,π].

Timed position RMSE is
\(\sqrt{T^{-1}\int\|p(t)-p_{ref}(t)\|^2dt}\) on the overlapping interval.
Both positions are linearly interpolated; the squared difference is integrated
analytically on each shared segment. No rigid fit, temporal shift or endpoint
extrapolation is applied. Timed heading uses unwrapped interpolation followed
by wrapped errors and trapezoidal integration. Time coverage is always reported.

Automatic evaluation freezes each initial Nav2 plan in its declared frame and
transforms measured odometry into that frame using the latest available TF.
Trace timestamps come from odometry message headers relative to the first
sample; each episode records `clock_origin_sim_s` to recover absolute simulation
time. Repeated/non-increasing pose stamps abort scoring rather than duplicate
a stale pose. TF and IMU remain asynchronous, not exactly time-synchronized. It exports the actual/reference
CSVs, config, metadata, per-episode scores and a combined summary. These are
**estimated tracking errors**, not motion-capture or simulator-world truth.
AMCL corrections can affect the estimated trajectory. Matching frames does not
by itself prove correct calibration/localization.

Always read success, endpoint error and final progress with RMSE. A stationary
robot on the path has zero path RMSE and zero completion. Partial timed overlap
can likewise look good, so coverage matters. Nearest-segment progress is
ambiguous on loops/intersections; use timestamped references for those cases.
Reported backtracking/travel distance are sampled estimates and can include
localization noise. Timing statistics over failed episodes are not successful
completion times; the summary provides successful-only means separately.

## Validation boundaries

The patch includes numerical metric/timing tests, a fake-transport execution of
the real environment step, controller regression tests, and a real SB3 PPO
update/save/load/resume test on a synthetic API environment. Those tests verify
software behavior, not robot dynamics or learning quality. ROS/Gazebo launch,
CUDA hardware execution, terrain learning and sim-to-real performance require
validation on the target machine. Follow the README's preflight → smoke run →
held-out baseline comparison sequence before claiming an improvement.
