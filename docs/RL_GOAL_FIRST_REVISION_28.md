# Goal-first wheel-control training and road grooves (revision 30)

The supplied TensorBoard run improved path tracking and challenge clearance
while successful arrivals disappeared. Revision 27 corrected premature
physics completion, timestamp freshness and lidar-visible goal decoration.
Revision 28 made the reward and exploration prioritize arrival. Revision 29
restores the established terrain after live inspection showed that scaling its
height weakened the pothole too much. Revision 30 expands lateral center
randomization to the wheel-center span and adds a traversable road groove.

## Why hazards belong along the route

In this fixed six-metre straight task, traversable hazards should exercise wheel
control on the route to the goal. The old generator was already near the route,
but a 12 cm-radius mound centered at y=0 could lie between drive wheels whose
centres are about 17.1 cm either side. A chassis crossing flag could pay for it
without a powered wheel crossing the mound.

The generator now uses the established near-route distribution again: up to
eight features are stratified through x=1.2–5.2 m and randomized each episode.
Each center is uniformly sampled from -17.1 cm to +17.1 cm, matching the left
and right powered-wheel centers. Objects stay fixed during an
episode. Five consecutive successes still add one hazard, up to the cap. The
six-phase, 100k-step schedule remains 6→5→4→3→2→1.

Adaptive terrain remains at full geometric height in every phase. The pothole
surrogate is again a 0.60 m basin with 30 mm relative relief and approximately
12.5-degree ramps; it is no longer reduced to 25% in phase 6. The low mound and
short adaptive cable also retain their proven geometry. Tall blocking posts
remain excluded. The phase cable still controls curriculum difficulty, and its
tilt sign is randomized every episode. The easiest phase now uses ±5 degrees
instead of a fixed 0-degree cable, so startup training also sees both wheel-first
crossing directions.

The fourth adaptive kind is a transverse road-groove surrogate. Two 90 mm
sloped shoulders rise 20 mm around a 100 mm-wide, 700 mm-long floor-level
channel. Both powered-wheel tracks can encounter the profile, while lateral
randomization changes which wheel meets it first and how centrally it crosses.
Because runtime SDF cannot subtract the existing hall floor, this represents a
groove as a relative drop in a locally raised road surface.

Challenge flags now use estimated swept powered-wheel footprints, including
heading, rather than a wide chassis-centre margin. A geometric crossing is
still not a force/contact measurement. Coordinates remain privileged reward
information and are not sent to the actor.

## Goal approach and reward

Progress is `10 × decrease in Euclidean distance to the actual goal`. Positive
progress is gated by alignment; increasing goal distance gets the full negative
term. This keeps a signal after the path projection reaches its end, where the
old remaining-arc-distance term became zero even while driving away.

The speed reference uses remaining along-path distance as well as endpoint
distance. A lateral miss or longitudinal overshoot no longer causes the command
to accelerate away. Inside the position circle with incorrect heading, a small
forward reference keeps differential residual steering enabled. Arrival still
requires the 10 cm circle and heading within 12 degrees. The forward-only task
ends with a -100 `goal_missed` failure after 30 cm longitudinal overshoot.
Deployment uses the same approach and overshoot logic as training.

The proportional slowdown distance is reduced from 1.0 m to 0.5 m. With the
initial speed multiplier, the previous approach spent much of the remaining
episode creeping toward the circle. A kinematic regression check ensures the
configured reference can finish the straight route before the 15 s target;
this is feasibility, not a guarantee under terrain, dynamics or steering error.

| Reward budget | Value |
|---|---:|
| Reaching the actual goal | +100 |
| Successful challenge-route arrival | up to +90 |
| Challenge entry / clearance during the episode | up to +2 / +8 |
| Arrival before the 15 s target | up to +50, based on time saved |
| Time spent | -0.05 per nominal 0.1 s step |
| Timeout, missed goal, collision and other failures | -100; off-path -75 |

Accuracy, slip, impact, effort and smoothness costs still apply. The total
successful challenge budget stays +100; most requires successful arrival.
On the default route, progress plus immediate challenge rewards is at most
+70 undiscounted before costs, below every failure penalty. This engineering
budget does not constitute a proof of convergence under PPO discounting.

## Policy and diagnostics

Keep the 300-value history observation, actor/critic architecture and independent
left/right residual action mapping. Initial Gaussian standard deviations become
`[0.20, 0.12, 0.05]` for speed, common torque and differential steering.
Previously all were 0.25. At the 2 Nm residual scale, steering noise alone had
about 1 Nm standard deviation in right-minus-left requested torque before
clipping; now it starts at about 0.2 Nm. This reduces initial wandering without
removing either wheel's action range. All three variances remain trainable and
are preserved on save/load. No scripted steering controller is added.

PPO learning rate, clipping, rollout size and KL protection are unchanged.
The evidence did not establish that those were causing the failure.

Watch success, timeout, `goal_missed_failure`, endpoint distance, final heading
and lateral drift alongside whole-episode reward totals. `policy/std_*` shows
each learned exploration scale. Each `episodes.jsonl` row also records the
actual terrain layout and height scale (now always 1.0), so poor outcomes can be traced to a
particular geometry rather than an averaged curve.

Use a new run, since reward, placement, crossing flags and action exploration
now differ from old checkpoints. Retain old logs as a comparison, not as a
compatible training continuation. Short live tests verify startup and software
behavior; longer held-out evaluations are still needed to establish learning
quality, finishing speed and robustness across all phases.

## Validation on 2026-09-22

- RL and controller tests: **140 passed**, with one unrelated Matplotlib
  Axes3D import warning. ROS packages rebuilt successfully.
- Mandatory 12-point preflight passed, followed by a fresh CUDA 2048-step
  rollout and PPO update. Run:
  `rl_runs/goal_first_validation/20260922-225931-351997`.
  Ten episodes completed: five successes, five timeouts, no off-path failures
  or collisions. This initial rollout was collected before the first update,
  not an evaluation of a converged policy.
- Every rollout episode cleared both challenges. Successful returns were
  approximately +237 to +239; failed returns were approximately -69 to -122.
  Episode reward components reconciled with returns within 1e-7. Maximum
  episode-clock accounting error was 1.82e-11 s; motion sample lag was at most
  0.022 s. The PPO update stopped early on its KL guard (about 0.05).
- Deterministic evaluation of that saved policy on seeds 101, 102 and 103
  gave one success and two timeouts. All cleared both challenges. With the
  original 1 m slowdown, the successful episode took 19.8 s; after the final
  0.5 m slowdown adjustment it took 15.4 s. The other two still timed out,
  about 23–24 cm from the goal, chiefly from lateral error. Returns with the
  final approach configuration were -94.85, +244.80 and -98.52 respectively.
  The model weights were held fixed for this inference comparison; the smoke
  checkpoint/config precedes the final slowdown adjustment. Gazebo dynamics
  and sensor timing mean this is not an exactly controlled paired trial.

These small tests establish executable training, valid reward accounting and
strict goal checking—not reliable success across the curriculum. The remaining
lateral misses need learned steering correction and longer evaluation. No
600k-step training run was started automatically. Start a fresh run with the
final configuration; do not resume the diagnostic checkpoint or older rewards.
