# Map and explain MPPI controller

> These three repos have an implementation of MPPI (a version of MPC). It builds on a version of RL (sharing the same cost, etc.) that was previously written.
>
> I want to better understand the MPPI implementation that was "written" by a student using AI help.
>
> Can you make a diagram of the inputs and outputs throughout the MPPI controller? I also want you to explain the code to me section by section or line by line if necessary to help me understand and debug if what he wrote is correct and reasonable. We are using it as a benchmark against RL to show benefits and weaknesses of each.

<details><summary>6 previous messages</summary>

> I’ll inspect the workspace to identify the three repositories and locate the MPPI/RL implementations, then trace the controller’s data flow and compare it against standard MPPI equations. Because this is a student robotics/controls codebase, I’ll keep the review evidence-based, call out assumptions and possible correctness issues, and avoid changing code unless you explicitly approve edits.
>
> <details><summary>Ran 8 commands</summary>
>
> - Ran `Get-ChildItem -Force | Select-Object Mode,Length,LastWriteTime,Name; rg --files -g '!*\.git/*' | Select-Object -First 300`
> - Ran `rg -n -i "mppi|model predictive|path integral|rollout|sample" baloo-control baloo-gym baloo-mujoco-sim -g '*.py' -g '*.md' -g '*.yaml' -g '*.xml'`
> - Ran `$i=1; Get-Content -LiteralPath 'baloo-mujoco-sim/src/baloo_mujoco_sim/controllers/mpc/mpc_controller.py' | ForEach-Object { '{0,4}: {1}' -f $i++, $_ }`
> - Ran `$i=1; Get-Content -LiteralPath 'baloo-mujoco-sim/src/baloo_mujoco_sim/controllers/mpc/mpc_controller_optuna.py' | ForEach-Object { '{0,4}: {1}' -f $i++, $_ }`
> - Ran `$i=1; Get-Content -LiteralPath 'baloo-control/nodes/MPC/mpc_controller_hardware.py' | ForEach-Object { '{0,4}: {1}' -f $i++, $_ }`
> - Ran `$i=1; Get-Content -LiteralPath 'baloo-control/nodes/MPC/mpc_controller_hardware_headless.py' | ForEach-Object { '{0,4}: {1}' -f $i++, $_ }`
> - Ran `$i=1; Get-Content -LiteralPath 'baloo-control/nodes/MPC/mpc_controller_hardware_README.md' | ForEach-Object { '{0,4}: {1}' -f $i++, $_ }`
> - Ran `Get-ChildItem -Directory | ForEach-Object { $p=$_.FullName; Write-Output "===== $($_.Name) ====="; if (Test-Path (Join-Path $p '.git')) { git -C $p status --short --branch } else { Write-Output 'no .git directory at repo root' } }`
>
> </details>
>
> I found the three repositories: `baloo-gym`, `baloo-mujoco-sim`, and `baloo-control`. The MPPI implementation exists in the MuJoCo simulation controller plus simulation-backed hardware controllers in `baloo-control`; the gym repository supplies the environment/reward and RL-side conventions. I’m now separating the variants so the diagram distinguishes “sim planning,” “hardware state injection,” and “actual hardware commands.”
>
> <details><summary>Ran 21 commands</summary>
>
> - Ran ``$repos=@('baloo-gym','baloo-mujoco-sim','baloo-control'); foreach($repo in $repos){ Write-Output "===== $repo ====="; git -c safe.directory="C:/Users/mdk28/Documents/git/baloo_help/$repo" status --short --branch; git -c safe.directory="C:/Users/mdk28/Documents/git/baloo_help/$repo" log -1 --date=iso --pretty=format:'%h %ad %s'; Write-Output "`n" }``
> - Ran ``$repos=@('baloo-gym','baloo-mujoco-sim','baloo-control'); foreach($repo in $repos){ $full=(Join-Path (Get-Location) $repo); Write-Output "===== $repo ====="; git -C $full -c safe.directory=$full status --short --branch; git -C $full -c safe.directory=$full log -1 --date=iso --pretty=format:'%h %ad %s'; Write-Output "`n" }``
> - Ran `$s=168; $e=390; $lines=Get-Content -LiteralPath 'baloo-mujoco-sim/src/baloo_mujoco_sim/controllers/mpc/mpc_controller_optuna.py'; for($j=$s;$j -le [Math]::Min($e,$lines.Count);$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
> - Ran `$s=1; $e=340; $lines=Get-Content -LiteralPath 'baloo-control/nodes/MPC/mpc_controller_hardware.py'; for($j=$s;$j -le [Math]::Min($e,$lines.Count);$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
> - Ran `$s=1; $e=340; $lines=Get-Content -LiteralPath 'baloo-control/nodes/MPC/mpc_controller_hardware_headless.py'; for($j=$s;$j -le [Math]::Min($e,$lines.Count);$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
> - Ran `$s=1; $e=260; $lines=Get-Content -LiteralPath 'baloo-gym/src/baloo_gym/wrappers/three_part_reward_wrapper.py'; for($j=$s;$j -le [Math]::Min($e,$lines.Count);$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
> - Ran `$s=1; $e=320; $lines=Get-Content -LiteralPath 'baloo-gym/src/baloo_gym/envs/baloo_v9.py'; for($j=$s;$j -le [Math]::Min($e,$lines.Count);$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
> - Ran `$s=1; $e=220; $lines=Get-Content -LiteralPath 'baloo-gym/src/baloo_gym/utils/action_spaces.py'; for($j=$s;$j -le [Math]::Min($e,$lines.Count);$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
> - Ran `$s=1; $e=180; $lines=Get-Content -LiteralPath 'baloo-gym/src/baloo_gym/utils/helpers.py'; for($j=$s;$j -le [Math]::Min($e,$lines.Count);$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
> - Ran `$lines=Get-Content -LiteralPath 'baloo-gym/src/baloo_gym/wrappers/three_part_reward_wrapper.py'; for($j=1;$j -le [Math]::Min(220,$lines.Count);$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
> - Ran `$lines=Get-Content -LiteralPath 'baloo-gym/src/baloo_gym/envs/baloo_v9.py'; for($j=1;$j -le [Math]::Min(210,$lines.Count);$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
> - Ran `$lines=Get-Content -LiteralPath 'baloo-mujoco-sim/src/baloo_mujoco_sim/controllers/mpc/mpc_controller_optuna.py'; for($j=1;$j -le [Math]::Min(170,$lines.Count);$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
> - Ran `$lines=Get-Content -LiteralPath 'baloo-control/nodes/MPC/mpc_controller_hardware.py'; for($j=148;$j -le [Math]::Min(280,$lines.Count);$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
> - Ran `$lines=Get-Content -LiteralPath 'baloo-control/nodes/MPC/mpc_controller_hardware.py'; for($j=378;$j -le [Math]::Min(514,$lines.Count);$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
> - Ran `$lines=Get-Content -LiteralPath 'baloo-gym/src/baloo_gym/wrappers/three_part_reward_wrapper.py'; for($j=220;$j -le [Math]::Min(400,$lines.Count);$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
> - Ran `$lines=Get-Content -LiteralPath 'baloo-gym/src/baloo_gym/envs/baloo_base.py'; for($j=1;$j -le [Math]::Min(260,$lines.Count);$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
> - Ran `$lines=Get-Content -LiteralPath 'baloo-control/nodes/MPC/mpc_controller_hardware_headless.py'; for($j=210;$j -le [Math]::Min(330,$lines.Count);$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
> - Ran `$lines=Get-Content -LiteralPath 'baloo-gym/src/baloo_gym/policies/open_loop_hugger.py'; for($j=1;$j -le [Math]::Min(260,$lines.Count);$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
> - Ran `$lines=Get-Content -LiteralPath 'baloo-control/nodes/MPC/mpc_controller_hardware_headless.py'; for($j=430;$j -le [Math]::Min(515,$lines.Count);$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
> - Ran `rg -n "def (get_box_position|get_box_quat|get_joint_angles|get_joint_vel|get_box_vel)|def set_(joint_pressure_commands|elevator_cmd|joint_angles|joint_velocities)|mj_forward" baloo-mujoco-sim/src/baloo_mujoco_sim/utils/baloo_mj_api.py; $lines=Get-Content 'baloo-mujoco-sim/src/baloo_mujoco_sim/utils/baloo_mj_api.py'; foreach($pair in @(@(1,180),@(180,380))){$s=$pair[0];$e=$pair[1]; for($j=$s;$j -le [Math]::Min($e,$lines.Count);$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }}`
> - Ran `$lines=Get-Content 'baloo-gym/src/baloo_gym/envs/baloo_base.py'; for($j=239;$j -le 340;$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
>
> </details>
>
> Several important review issues are already emerging, including at least one action-scaling error in the rollout model and benchmark mismatches between the MPPI cost and the RL wrapper. I’m checking these against the shared action mapping and reward code before labeling them as bugs, because the distinction between “planning simulation,” “environment execution,” and “hardware command conversion” matters here.
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `$a=Get-Content 'baloo-control/nodes/MPC/mpc_controller_hardware.py'; $b=Get-Content 'baloo-control/nodes/MPC/mpc_controller_hardware_headless.py'; Write-Output "hardware lines=$($a.Count), headless lines=$($b.Count)"; Compare-Object $a $b | Select-Object -First 80 | Format-Table -AutoSize; rg -n "w_guide|W_GUIDE|box_lifted|R_LIFT|R_TIP|W_HEIGHT|apply_normalized_action|perturbations|best_action|mj_setState|mj_forward|object_xpos|object_zrotation|copy_baseline|dont_drop" baloo-mujoco-sim/src/baloo_mujoco_sim/controllers/mpc baloo-control/nodes/MPC baloo-gym/src/baloo_gym/wrappers/three_part_reward_wrapper.py baloo-gym/src/baloo_gym/envs/baloo_v9.py baloo-gym/src/baloo_gym/utils/action_spaces.py`
> - Ran `$lines=Get-Content 'baloo-gym/src/baloo_gym/utils/observation_spaces.py'; for($j=1;$j -le 260;$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
> - Ran `rg -n "class StateObservationObjectOnly|shape =" baloo-gym/src/baloo_gym/utils/observation_spaces.py | Select-Object -Last 20; $lines=Get-Content 'baloo-gym/src/baloo_gym/utils/observation_spaces.py'; for($j=350;$j -le 510;$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
> - Ran `$lines=Get-Content 'baloo-gym/src/baloo_gym/utils/observation_spaces.py'; for($j=1319;$j -le 1435;$j++){ '{0,4}: {1}' -f $j,$lines[$j-1] }`
>
> </details>
</details>

I reviewed all three repositories without modifying them. Their working trees were clean at the start.

The high-level structure is reasonable, but the current implementation should not yet be treated as a trustworthy MPPI-vs-RL benchmark. There are several important correctness and fairness issues.

## Controller data-flow diagram

```text
                         Configuration
              box size, mass, pose, MPPI parameters
                              │
                              ▼
                 Open-loop nominal policy
                 OpenLoopHuggerPolicy
                              │
                              ▼
              nominal_actions[t:t+H]  shape H × 13
                              │
                              │
Current state ────────────────┘
MuJoCo state / Vive / joint sensors
          │
          ▼
   full MuJoCo state vector
          │
          ▼
 ┌──────────────────────────────────────────────┐
 │ Sample N perturbed action sequences           │
 │                                              │
 │ ε ~ Normal(0, σ)                             │
 │ candidate[i] = nominal_window + ε[i]         │
 │ shape: N × H × 13                            │
 └──────────────────────────────────────────────┘
          │
          ▼
 ┌──────────────────────────────────────────────┐
 │ Parallel MuJoCo rollout workers              │
 │                                              │
 │ For every candidate and every horizon step:  │
 │   normalized action                         │
 │        ↓                                     │
 │   pressure/elevator conversion               │
 │        ↓                                     │
 │   MuJoCo control inputs                      │
 │        ↓                                     │
 │   mj_step                                    │
 │        ↓                                     │
 │   guide cost accumulation                    │
 │                                              │
 │ Terminal state → task reward                 │
 └──────────────────────────────────────────────┘
          │
          ▼
       costs[i]
          │
          ▼
 Softmin weights:
 w[i] = exp(-(cost[i] - min_cost) / λ)
          │
          ▼
 Weighted first action:
 best_action = Σ w[i] candidate[i,0]
          │
          ├──────────── Simulation:
          │             BalooV9.step()
          │
          └──────────── Hardware:
                        filter pressures
                        convert to 25 commands
                        send to elevator and arms
                              │
                              ▼
                      Next measured state
                      and next MPPI iteration
```

The action dimensions are:

```text
u[0]      elevator command
u[1:3]    left joint 0 x/y pressure deltas
u[3:5]    left joint 1 x/y pressure deltas
u[5:7]    left joint 2 x/y pressure deltas
u[7:9]    right joint 0 x/y pressure deltas
u[9:11]   right joint 1 x/y pressure deltas
u[11:13]  right joint 2 x/y pressure deltas
```

The MPPI action is 13-dimensional. It is eventually converted to 25 physical commands: one elevator command plus 24 chamber pressures.

## What the algorithm is doing

The core implementation is in [`mpc_controller.py`](./baloo-mujoco-sim/src/baloo_mujoco_sim/controllers/mpc/mpc_controller.py:19).

For each control step:

1. It obtains the current MuJoCo state.
2. It takes a window of the fixed open-loop action trajectory.
3. It adds Gaussian noise to create candidate trajectories.
4. It simulates every candidate in parallel.
5. It assigns each candidate a cost.
6. It converts costs into softmin weights.
7. It averages the first action of all candidates using those weights.
8. It executes only that first action.
9. It repeats from the new state.

Mathematically:

\[
u_i[k] = u_{\text{nom}}[k] + \epsilon_i[k]
\]

\[
J_i =
\sum_k
\left[
-w_g \exp\left(-\frac{1}{2}\|u_i[k]-u_{\text{nom}}[k]\|^2\right)
\right]
-
R_{\text{terminal},i}
\]

\[
w_i =
\frac{\exp(-(J_i-J_{\min})/\lambda)}
{\sum_j \exp(-(J_j-J_{\min})/\lambda)}
\]

\[
u_{\text{execute}} =
\sum_i w_i u_i[0]
\]

That weighted-candidate update is equivalent to the usual weighted-noise update:

\[
u_{\text{execute}}
=
u_{\text{nom}}[0]+\sum_i w_i\epsilon_i[0].
\]

So the basic sampling idea is valid.

However, this is closer to “guided sampling-based MPC” than a full canonical MPPI implementation. The code:

- does not warm-start from the previous optimized sequence;
- does not shift an optimized control sequence forward;
- uses a fixed open-loop nominal trajectory;
- uses a terminal task reward rather than a running state cost;
- does not explicitly include the usual MPPI control-noise cost formulation.

That can still be a reasonable controller, but it should be described accurately.

## Section-by-section explanation

### 1. Hyperparameters

The current hardware values are defined in [`mpc_controller_hardware.py`](./baloo-control/nodes/MPC/mpc_controller_hardware.py:48):

- `N_SAMPLES = 31`: candidate trajectories per control step.
- `HORIZON = 16`: each candidate predicts 16 steps, or 0.8 seconds at 20 Hz.
- `SIGMA = 0.275`: normalized-action perturbation magnitude.
- `W_GUIDE = 0.185`: preference for staying near the nominal action.
- `W_HEIGHT = 157.45`: continuous reward for increasing box height.
- `LAMBDA_TEMP = 3.727`: softmin temperature.

Larger `lambda` makes the controller average more candidates. Smaller `lambda` makes it behave more like choosing the lowest-cost candidate.

### 2. Worker model initialization

[`worker_init`](./baloo-control/nodes/MPC/mpc_controller_hardware.py:213) creates one MuJoCo model per multiprocessing worker.

This is a sensible performance optimization. Each worker receives a copied state and simulates independently.

### 3. Tipping and lifting tests

[`box_tipped`](./baloo-control/nodes/MPC/mpc_controller_hardware.py:227) computes the angle between the box’s local vertical axis and world vertical. More than 80 degrees is classified as tipped.

The quaternion conversion appears consistent with the project’s convention: the project stores MuJoCo quaternions as `[w, x, y, z]`, while SciPy expects `[x, y, z, w]`.

There is an important inconsistency in [`box_lifted`](./baloo-control/nodes/MPC/mpc_controller_hardware.py:233):

```python
return get_box_position(model, data)[2] > threshold
```

This uses an absolute world height of `0.5 m`.

The RL reward instead uses:

```python
box_height > self.initial_height + 0.5
```

from [`three_part_reward_wrapper.py`](./baloo-gym/src/baloo_gym/wrappers/three_part_reward_wrapper.py:151).

Since the box initially starts at approximately half its height, these are not equivalent. For a 0.6 m box, the MPPI condition may trigger after only about 0.2 m of lift, while the RL condition requires 0.5 m.

### 4. Guide cost

[`guide_cost`](./baloo-control/nodes/MPC/mpc_controller_hardware.py:236) gives a negative cost when the candidate action is close to the nominal action.

That is mathematically reasonable as an action-prior reward:

```python
-W_GUIDE * exp(-0.5 * ||action - nominal||²)
```

But there is a serious bug in the Optuna version. [`run_episode`](./baloo-mujoco-sim/src/baloo_mujoco_sim/controllers/mpc/mpc_controller_optuna.py:170) accepts `w_guide`, and Optuna tunes it, but [`guide_cost`](./baloo-mujoco-sim/src/baloo-mujoco-sim/controllers/mpc/mpc_controller_optuna.py:96) uses the global `W_GUIDE`.

Therefore, the tuned `w_guide` value has no effect. The Optuna search is optimizing a parameter that is not actually used.

### 5. Task reward

The rollout computes:

```python
total_cost -= task_reward(...)
```

This correctly converts a positive reward into a lower cost.

The current hardware version combines:

- terminal height progress;
- a lift bonus;
- a tipping penalty.

That is a reasonable objective, although it is only evaluated after the complete horizon. A trajectory that tips halfway through the horizon but recovers before the final step may avoid being penalized.

Also, the hardware planner does not terminate a rollout when the box tips. It continues simulating after failure.

### 6. Action conversion

This is the most serious implementation issue.

The normal environment conversion is defined in [`action_spaces.py`](./baloo-gym/src/baloo_gym/utils/action_spaces.py:79):

```python
normalized -1 → elevator -900
normalized +1 → elevator 0
```

The actual environment uses this correct mapping in [`baloo_v9.py`](./baloo-gym/src/baloo_gym/envs/baloo_v9.py:140).

But the MPPI rollout uses this in [`apply_normalized_action`](./baloo-control/nodes/MPC/mpc_controller_hardware.py:247):

```python
set_elevator_cmd(model, data, (a[0] + 1) / 2 * (-900))
```

This maps:

```text
a[0] = -1 → 0
a[0] = +1 → -900
```

That is reversed.

Therefore, the planner evaluates candidates with the elevator moving in the opposite direction from the actual environment and hardware. In the current hardware loop the elevator is forced to follow the nominal action, so this still corrupts every rollout’s simulated dynamics.

The correct equivalent formula should map `-1` to `-900` and `+1` to `0`.

### 7. Nominal trajectory

The nominal trajectory is generated by repeatedly running [`OpenLoopHuggerPolicy`](./baloo-gym/src/baloo_gym/policies/open_loop_hugger.py:77).

This policy is mostly time/state-machine based:

- `APPROACH`
- `GRASP`
- `LIFT`

The transition to `LIFT` does not verify that the box was actually grasped. It mainly verifies that the planned sequence has completed and the elevator is at the expected position.

Thus, the handoff to MPPI is a planned-time handoff, not a measured-success handoff.

### 8. Candidate sampling

The current hardware code samples:

```python
perturbations.shape = (31, 16, 13)
```

from [`mpc_controller_hardware.py`](./baloo-control/nodes/MPC/mpc_controller_hardware.py:473).

The elevator noise is explicitly set to zero:

```python
perturbations[:, :, 0] = 0.0
```

That is a valid design choice if the elevator should remain nominal. It means MPPI only controls the 12 arm pressure-delta dimensions.

### 9. Rollout costs and weighted action

The weighted action is computed in [`mpc_controller_hardware.py`](./baloo-control/nodes/MPC/mpc_controller_hardware.py:480).

The numerical stabilization trick:

```python
costs - costs.min()
```

is good. It prevents numerical underflow in the exponential.

The selected action is not explicitly clipped before the rollout update in the hardware version. It is clipped later during hardware conversion, so the planner’s evaluated action and executed action are not always identical.

### 10. Hardware state injection

The hardware-specific flow is:

```text
Vive box/base poses
        ↓
frame transformations
        ↓
box pose in robot-base frame
        ↓
MuJoCo box qpos/qrot

hardware joint angles/velocities
        ↓
MuJoCo arm joint state
```

This occurs in [`inject_vive_box_into_sim`](./baloo-control/nodes/MPC/mpc_controller_hardware.py:340) and [`inject_hw_joints_into_sim`](./baloo-control/nodes/MPC/mpc_controller_hardware.py:359).

The Vive twist callbacks are stored, but they are never used in the state injection. The box velocity is effectively reset to zero:

```python
data.qvel[vadr:vadr+6] = 0.0
```

That makes the planner’s initial box velocity inconsistent with the real box motion.

### 11. Hardware filtering mismatch

The actual hardware action is low-pass filtered before sending:

```python
pressure_actions = lpf.apply_filter(best_action[1:])
```

However, the MPPI rollout worker directly applies the candidate pressure commands without the same low-pass filter.

The MuJoCo environment step applies its own filter, but the worker rollouts do not. Therefore:

```text
predicted dynamics ≠ executed hardware dynamics
```

This can make MPPI believe pressure changes take effect faster than they actually do.

The same issue exists in the simulation controller: rollout actions are applied directly, while `BalooV9.step()` filters them.

## RL cost comparison

The RL reward implementation is in [`three_part_reward_wrapper.py`](./baloo-gym/src/baloo_gym/wrappers/three_part_reward_wrapper.py:125).

The MPPI code claims to match the guided RL reward, but the current Optuna episode uses:

```python
ThreePartRewardWrapper(env, reward_selection=['dont_drop'])
```

from [`mpc_controller_optuna.py`](./baloo-mujoco-sim/src/baloo_mujoco_sim/controllers/mpc/mpc_controller_optuna.py:191).

That does not enable the RL `copy_baseline` action-guidance reward. So the actual MPPI objective and actual RL wrapper objective are not the same.

There is also a sign bug in the RL tipping reward:

```python
box_fell_over_reward = -2
reward -= box_fell_over_reward
```

Since `box_fell_over_reward` is already negative, this adds `+2` reward when the box tips. The MPPI code correctly treats tipping as a penalty.

The `dont_drop` reward also contains duplicated logic and computes the drop penalty after setting the off-floor counter to zero, so the intended penalty is effectively zero.

## Most important issues to fix or verify

| Priority | Issue |
|---|---|
| Critical | Elevator mapping is reversed inside MPPI rollouts. |
| Critical | MPPI lift threshold is absolute height, while RL uses height relative to initial height. |
| Critical | MPPI and RL do not currently use the same reward. |
| Critical | Optuna tunes `w_guide`, but that parameter is ignored. |
| Critical | Optuna accepts box position/rotation but does not pass them into the environments. |
| High | Rollouts omit the hardware/environment low-pass filter. |
| High | Hardware box velocity is discarded despite velocity topics being subscribed. |
| High | Current hardware/simulation environments are not wrapped for termination, so success/failure does not stop execution. |
| Medium | Rollouts only penalize the terminal state, not intermediate tipping. |
| Medium | The nominal trajectory is fixed rather than warm-started and shifted. |
| Medium | Optuna uses an unseeded random generator, reducing repeatability. |
| Medium | The old simulation controller contains a hard-coded absolute import path. |

## Bottom line

The architecture is sensible:

- nominal policy;
- Gaussian perturbations;
- parallel model rollouts;
- softmin weighting;
- first-action execution;
- receding-horizon repetition.

But the current code has enough model, reward, and action-conversion mismatches that an observed MPPI-vs-RL performance difference could be caused by implementation artifacts rather than algorithmic benefits.

Before using it as a benchmark, I would first make the rollout action mapping, lift criterion, reward function, termination behavior, and filtering identical between:

1. the MPPI rollout model;
2. the actual simulation environment;
3. the hardware command path;
4. the RL benchmark environment.

I did not make any code changes.
