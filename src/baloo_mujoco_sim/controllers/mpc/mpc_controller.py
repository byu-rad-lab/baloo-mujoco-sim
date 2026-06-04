# MPPI controller guided by openloop
import numpy as np
import mujoco
import time
from multiprocessing import Pool
from baloo_gym.envs.baloo_v9 import BalooV9
# it uses ThreePartRewardWrapper for visual purposes only (green box when lifting)
from baloo_gym.wrappers.three_part_reward_wrapper import ThreePartRewardWrapper
from baloo_gym.policies.open_loop_hugger import OpenLoopHuggerPolicy
from baloo_mujoco_sim.utils.baloo_mj_api import (
    get_box_position, get_box_quat,
    get_link_position,
    set_elevator_cmd, set_joint_pressure_commands
)
from scipy.spatial.transform import Rotation as R
import baloo_mujoco_sim

N_SAMPLES     = 32
HORIZON       = 8
SIGMA         = 0.15
N_OPEN_LOOP   = 50
W_GUIDE       = 0.01
W_ARM_DIST    = 3.0
R_LIFT        = 10.0
R_TIP         = -2.0
OBJECT_SIZE   = (0.3, 0.3, 0.6)
OBJECT_MASS   = 10      # 0.5 to 10.0 range
CTRL_TIMESTEP = 0.05
N_WORKERS     = 16
LAMBDA_TEMP   = 1.0

_worker_model = None

def worker_init(xml_path):
    global _worker_model
    _worker_model = mujoco.MjModel.from_xml_path(xml_path)

def box_tipped(model, data):
    quat  = get_box_quat(model, data)
    rot   = R.from_quat(np.roll(quat, -1))
    box_z = rot.apply([0, 0, 1])
    return np.degrees(np.arccos(np.clip(np.dot(box_z, [0, 0, 1]), -1, 1))) > 80

def box_lifted(model, data, threshold=0.5):
    return get_box_position(model, data)[2] > threshold

def guide_cost(action, nominal_action):
    diff = action - nominal_action
    return -W_GUIDE * np.exp(-0.5 * np.dot(diff, diff))

def arm_dist_cost(model, data):
    box_pos   = get_box_position(model, data)
    left_tip  = get_link_position(model, data, 'left',  1)
    right_tip = get_link_position(model, data, 'right', 1)
    return W_ARM_DIST * (np.linalg.norm(left_tip - box_pos) + np.linalg.norm(right_tip - box_pos))

def task_reward(model, data):
    if box_tipped(model, data): return R_TIP
    if box_lifted(model, data): return R_LIFT
    return 0.0

def apply_normalized_action(model, data, a):
    a = np.clip(a, -1, 1)
    pressure_deltas = a[1:] * 150
    set_elevator_cmd(model, data, (a[0] + 1) / 2 * (-900))
    for side_idx, side in enumerate(['left', 'right']):
        for j in range(3):
            idx = side_idx * 6 + j * 2
            set_joint_pressure_commands(model, data, side, j, np.array([
                150 + pressure_deltas[idx],
                150 - pressure_deltas[idx],
                150 + pressure_deltas[idx + 1],
                150 - pressure_deltas[idx + 1],
            ]))

# cost per sample:
#   guide_cost    : soft penalty for deviating from nominal trajectory (all steps)
#   arm_dist_cost : penalize arms far from box (approach phase only, steps 49-99)
#   task_reward   : subtracted because lower cost = better (-10 lift, +2 tip)
def rollout_worker(args):
    global _worker_model
    data_state, action_seq, nominal_seq, step_idx = args
    data = mujoco.MjData(_worker_model)
    mujoco.mj_setState(_worker_model, data, data_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    total_cost = 0.0
    for t in range(len(action_seq)):
        apply_normalized_action(_worker_model, data, action_seq[t])
        mujoco.mj_step(_worker_model, data)
        total_cost += guide_cost(np.clip(action_seq[t], -1, 1), nominal_seq[t])
    total_cost -= task_reward(_worker_model, data)
    if 49 <= step_idx < 99:
        total_cost += arm_dist_cost(_worker_model, data)
    return total_cost

def run_mpc():
    xml_path = str(baloo_mujoco_sim.XML_PATH)

    env = BalooV9(
        render_mode='human',
        camera_name='frontcam',
        ctrl_timestep=CTRL_TIMESTEP,
        render_width=640,
        render_height=480,
        randomize_initial_height=False,
        randomize_object_size=False,
        randomize_object_mass=False,
        object_size=OBJECT_SIZE,
        object_mass=OBJECT_MASS,
    )

    # box changes color based on weight and if lifting thanks to ThreePartRewardWrapper
    env = ThreePartRewardWrapper(env, reward_selection=['dont_drop'])   


    obs, _ = env.reset()
    env.render()
    viewer = env.unwrapped.mujoco_renderer._viewers.get('human')
    if viewer:
        viewer.cam.azimuth   = -102.5
        viewer.cam.elevation = -43.8
        viewer.cam.distance  = 3.0
        viewer.cam.lookat[0] = 0.0
        viewer.cam.lookat[1] = 0.0
        viewer.cam.lookat[2] = 0.5

    nom_policy = OpenLoopHuggerPolicy(N=N_OPEN_LOOP)
    max_steps  = 2000
    nom_env = BalooV9(render_mode=None, camera_name='frontcam', ctrl_timestep=CTRL_TIMESTEP,
        randomize_initial_height=False, randomize_object_size=False,
        randomize_object_mass=False, object_size=OBJECT_SIZE, object_mass=OBJECT_MASS)
    nom_obs, _ = nom_env.reset()
    nominal_actions = []
    for _ in range(max_steps):
        a, _ = nom_policy.predict(nom_obs)
        nominal_actions.append(a.copy())
        nom_obs, _, t, tr, _ = nom_env.step(a)
    nom_env.close()
    nominal_actions = np.array(nominal_actions)
    print(f"OBJECT_SIZE={OBJECT_SIZE} OBJECT_MASS={OBJECT_MASS}kg")
    print(f"N_SAMPLES={N_SAMPLES} HORIZON={HORIZON} WORKERS={N_WORKERS}")
    print("Starting MPC loop...")

    done = False
    step = 0
    rng  = np.random.default_rng(42)
    t0   = time.time()

    with Pool(N_WORKERS, initializer=worker_init, initargs=(xml_path,)) as pool:
        while not done and step < max_steps:
            model = env.unwrapped.model
            data  = env.unwrapped.data

            state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            data_state = np.zeros(state_size)
            mujoco.mj_getState(model, data, data_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)

            t_end      = min(step + HORIZON, len(nominal_actions))
            nom_window = nominal_actions[step:t_end]

            perturbations = rng.normal(0, SIGMA, size=(N_SAMPLES, len(nom_window), 13))
            candidates    = nom_window[None] + perturbations

            args  = [(data_state, candidates[i], nom_window, step) for i in range(N_SAMPLES)]
            costs = np.array(pool.map(rollout_worker, args))

            weights = np.exp(-(costs - costs.min()) / LAMBDA_TEMP)
            weights /= weights.sum()
            best_action = np.sum(weights[:, None] * candidates[:, 0, :], axis=0)
            obs, reward, terminated, truncated, info = env.step(best_action)
            env.render()
            done = terminated or truncated
            step += 1

            if step % 5 == 0:
                elapsed = time.time() - t0
                print(f"step {step:3d} | cost={costs.min():.3f} | "
                      f"box_z={get_box_position(model,data)[2]:.3f} | {step/elapsed:.1f} steps/sec")

    print(f"Done at step {step}.")
    env.close()

if __name__ == '__main__':
    run_mpc()
