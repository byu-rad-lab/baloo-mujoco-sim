import mujoco
from mrac.ManipulatorMRACRBF import ManipulatorMRACRBF
import numpy as np
import mujoco.viewer as viewer
import time
from baloo_mujoco_sim.utils.baloo_mj_api import get_joint_angles, get_joint_vel, set_joint_pressure_commands, get_joint_pressures, disable_gravity, set_mocap_pose
from baloo_mujoco_sim.utils.mjData_plotter import MjDataPlotter
import baloo_mujoco_sim as baloo_mj

from scipy.spatial.transform import Rotation as R
from continuum_kinematics_py import ContinuumKinematics

np.set_printoptions(precision=3, suppress=True)

#load baloo.xml file
model = mujoco.MjModel.from_xml_path(baloo_mj.XML_PATH)
data = mujoco.MjData(model)

# des_pos = np.array([-.4, .7, .7])
# des_quat = np.array([1, 0, 0, 0])

# des_R = R.from_quat(np.moveaxis(des_quat, 0, -1)).as_matrix()
# set_mocap_pose(model, data, "left_ee_mocap", des_pos, des_quat)

disable_gravity(model)

controller = ManipulatorMRACRBF(
    num_gen_coords=6,
    numberOfRBFCenters=
    10,  #number doesn't seem to matter too much for performance, but more smooths out the control signal to a point.
    RBFmins=np.array([-1] * 6 * 4) * 1.0,
    RBFmaxes=np.array([1] * 6 * 4) * 1.0,
    zeta=np.array([1] * 6),
    time_constant=np.array(
        [1] * 6
    ),  #this needs to be tuned aware of underlying dynamics, otherwise smoothing effect is lost. Optional if you have desired trajecotyr already.
    Lambda=10 * .5,  #lambda*KD is weight on position error qdes - q
    Gamma=10 *
    1,  #don't love having this so high, its kind of like the itegrator
    KD=1 * 2,  #weight on velocity error, too high causes chattering
    ctrl_dt=model.opt.timestep,
)

#notes: PD part too low  = unstable, to high = vibrations.

plotter = MjDataPlotter(30, model.opt.timestep)
# plotter.show()

first_time = True


def q_des_function_generator(t):
    # # step commands to 5 different places, 10 seconds at each place
    if t < 100:
        test = np.asarray([0.7, -0.5, -0.25, 0.4, -0.7, 0.5])
    elif t < 20:
        test = np.asarray([0.2, -0.1, -0.6, 0.4, -0.4, 0.3])
    elif t < 30:
        test = np.asarray([0.5, 0.5, 0.3, 0.3, 0.6, 0.4])
    elif t < 40:
        test = np.asarray([-0.7, 0.5, 0.25, -0.4, 0.7, 0.5])
    elif t < 50:
        test = np.asarray([-0.2, 0.1, 0.6, -0.4, 0.4, -0.3])
    elif t < 60:
        test = np.asarray([0.0, -0.0, .6, 0.6, -0.25, 0.8])
    elif t < 70:
        test = np.asarray([-.6, -0.3, -.6, -0.6, -0.5, -0.2])
    elif t < 80:
        test = np.asarray([0.0, -0.0, .0, -0.4, -0.25, 0.])
    elif t < 90:
        test = np.asarray([0.7, -0.4, .6, -0.6, -0.25, 0.8])
    elif t < 100:
        test = np.asarray([-0.4, .5, -.4, 0.2, 0.5, -0.8])
    elif t < 110:
        test = np.asarray([0.6, .3, .0, 0.6, -0.25, 0.6])
    else:
        test = np.zeros(6)

    # test = np.zeros(6)

    return test, None, None


def left_ee_des(t):
    if t < 10:
        des_pos = np.array([-.4, .7, .7])
    elif t < 20:
        des_pos = np.array([-.8, 1.0, 1])
    elif t < 30:
        des_pos = np.array([-0, .8, .8])
    elif t < 40:
        des_pos = np.array([-.4, 1.0, 1.2])
    else:
        des_pos = np.array([-.4, 1.4, .7])

    return des_pos


def q_des_sine_wave(t):
    t = t / (2 / np.pi)
    #sine wave commands
    qdes = np.asarray([
        0.5 * np.sin(t), 0.5 * np.cos(t), 0.5 * np.sin(t), 0.5 * np.cos(t),
        0.5 * np.sin(t), 0.5 * np.cos(t)
    ])
    qd_des = np.asarray([
        0.5 * np.cos(t), -0.5 * np.sin(t), 0.5 * np.cos(t), -0.5 * np.sin(t),
        0.5 * np.cos(t), -0.5 * np.sin(t)
    ])

    qdd_des = np.asarray([
        -0.5 * np.sin(t), -0.5 * np.cos(t), -0.5 * np.sin(t), -0.5 * np.cos(t),
        -0.5 * np.sin(t), -0.5 * np.cos(t)
    ])

    return qdes, qd_des, qdd_des


time_hist = []
u0_hist = []
u1_hist = []
u2_hist = []
v0_hist = []
v1_hist = []
v2_hist = []

u0_cmd = []
u1_cmd = []
u2_cmd = []
v0_cmd = []
v1_cmd = []
v2_cmd = []

ud0_cmd = []
ud1_cmd = []
ud2_cmd = []
vd0_cmd = []
vd1_cmd = []
vd2_cmd = []

ud0_hist = []
ud1_hist = []
ud2_hist = []
vd0_hist = []
vd1_hist = []
vd2_hist = []

theta_hist = []
s_hist = []

tau_hist = []
tau_ff_hist = []
tau_pd_hist = []

pressure_cmd_hist = []
pressure_hist = []
qd_ref_hist = []


def torque2pressures(torque):
    # convert torque to pressures
    p0 = 200 + torque
    p1 = 200 - torque

    return p0, p1


# test = ContinuumKinematics(67.5)

solve_times = []

with viewer.launch_passive(model, data) as viewer:
    start = time.time()

    while viewer.is_running():
        # if first_time:
        #     input("Press Enter to continue...")
        #     first_time = False
        step_start = time.time()

        q0 = get_joint_angles(model, data, "left", 0)
        q1 = get_joint_angles(model, data, "left", 1)
        q2 = get_joint_angles(model, data, "left", 2)

        q0dot = get_joint_vel(model, data, "left", 0)
        q1dot = get_joint_vel(model, data, "left", 1)
        q2dot = get_joint_vel(model, data, "left", 2)

        q = np.hstack([q0, q1, q2]).squeeze()
        qdot = np.hstack([q0dot, q1dot, q2dot]).squeeze()

        # q_des = np.asarray([0.7, -0.5, -0.25, 0.4, -0.7, 0.5])
        qd_des = None
        qdd_des = None
        q_des, qd_des, qdd_des = q_des_function_generator(data.time)
        # q_des, qd_des, qdd_des = q_des_sine_wave(data.time)

        # des_pos = left_ee_des(data.time)
        # set_mocap_pose(model, data, "left_ee_mocap", des_pos, des_quat)

        # q_des, iterations = test.solveDampedPseudoInvIK(
        #     "LEFT", q0, q1, q2, des_pos, des_R, .01, 100, .05, False)

        if data.time > 0:
            start = time.time()
            tau, s, theta_hat, tau_ff, tau_pd, q_des, qdot_des, qddot_des = controller.solve_for_next_u(
                q, qdot, q_des, qd_des, qdd_des)
            
            solve_times.append(time.time() - start)
            print(f"Time to solve: {time.time() - start}")

            # print(f"norm of s: {np.linalg.norm(s)}")

            # stiffness feedforward compensation, add torque required to get to desired position to input signal

            # tau_stiff_ff = 100 * q_des
            tau_stiffness = 100 * q_des
            # tau = tau + tau_stiffness

            # print(f"tau_stiffness ratio: {np.linalg.norm(tau_stiffness)/np.min(np.linalg.eigvals(controller.Kd))}")

            # if data.time > 30:
            #     tau = tau + tau_pd  #do only the feedforward control

            # print(q_des - q)
            # print(s)
            # tau = tau * 1e3
            # print(f"\n\ntau: {tau}")
            # print(data.time)
            # print(f"s: {s}")
            p_cmds = []
            ps = []
            for i in range(3):
                p0, p1 = torque2pressures(tau[i * 2])
                p2, p3 = torque2pressures(tau[(2 * i) + 1])
                p_cmds.append([p0, p1, p2, p3])
                set_joint_pressure_commands(model, data, "left", i,
                                            [p0, p1, p2, p3])
                ps.append(get_joint_pressures(model, data, "left", i))

            pressure_cmd_hist.append(p_cmds)
            pressure_hist.append(ps)

            time_hist.append(data.time)
            u0_hist.append(q[0])
            v0_hist.append(q[1])
            u1_hist.append(q[2])
            v1_hist.append(q[3])
            u2_hist.append(q[4])
            v2_hist.append(q[5])

            u0_cmd.append(q_des[0])
            v0_cmd.append(q_des[1])
            u1_cmd.append(q_des[2])
            v1_cmd.append(q_des[3])
            u2_cmd.append(q_des[4])
            v2_cmd.append(q_des[5])

            # ud0_cmd.append(qd_des[0])
            # vd0_cmd.append(qd_des[1])
            # ud1_cmd.append(qd_des[2])
            # vd1_cmd.append(qd_des[3])
            # ud2_cmd.append(qd_des[4])
            # vd2_cmd.append(qd_des[5])

            # ud0_hist.append(qdot[0])
            # vd0_hist.append(qdot[1])
            # ud1_hist.append(qdot[2])
            # vd1_hist.append(qdot[3])
            # ud2_hist.append(qdot[4])
            # vd2_hist.append(qdot[5])

            theta_hist.append(theta_hat)
            s_hist.append(s)
            tau_ff_hist.append(tau_ff)
            tau_pd_hist.append(tau_pd)
            tau_hist.append(tau)

            # qd_ref_hist.append(qd_ref)

        # data.ctrl = np.array([p0, p1, p2, p3])
        # print(theta_hat.shape)

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(model, data)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # start = time.time()
        # plotter.update(
        #     model, data, {
        #         "joint0_angle0_cmd": q_des[0],
        #         "joint0_angle1_cmd": q_des[1],
        #         "joint1_angle0_cmd": q_des[2],
        #         "joint1_angle1_cmd": q_des[3],
        #         "joint2_angle0_cmd": q_des[4],
        #         "joint2_angle1_cmd": q_des[5],
        #     })
        # print(time.time() - start)

        # Rudimentary time keeping, will drift relative to wall clock.
        # print(time.time() - step_start)
        # time_until_next_step = model.opt.timestep - (time.time() - step_start)
        # if time_until_next_step > 0:
        #     time.sleep(time_until_next_step)

        if data.time > 120:
            plotter.close()
            viewer.close()

    import matplotlib.pyplot as plt

    #make figure with 3x1 subplots to plot u0 and v0 with their commands on each plot
    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.suptitle("Adaptive Control of Baloo")
    axs[0].plot(time_hist, u0_hist, label="u0")
    axs[0].plot(time_hist, u0_cmd, "--", label="u0_cmd")
    axs[0].plot(time_hist, v0_hist, label="v0")
    axs[0].plot(time_hist, v0_cmd, "--", label="v0_cmd")
    axs[0].set_title("Joint 0")
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(time_hist, u1_hist, label="u1")
    axs[1].plot(time_hist, u1_cmd, "--", label="u1_cmd")
    axs[1].plot(time_hist, v1_hist, label="v1")
    axs[1].plot(time_hist, v1_cmd, "--", label="v1_cmd")
    axs[1].set_title("Joint 1")
    axs[1].grid()
    axs[1].legend()

    axs[2].plot(time_hist, u2_hist, label="u2")
    axs[2].plot(time_hist, u2_cmd, "--", label="u2_cmd")
    axs[2].plot(time_hist, v2_hist, label="v2")
    axs[2].plot(time_hist, v2_cmd, "--", label="v2_cmd")
    axs[2].set_title("Joint 2")
    axs[2].grid()
    axs[2].legend()

    u0_error = np.array(u0_cmd) - np.array(u0_hist)
    v0_error = np.array(v0_cmd) - np.array(v0_hist)
    u1_error = np.array(u1_cmd) - np.array(u1_hist)
    v1_error = np.array(v1_cmd) - np.array(v1_hist)
    u2_error = np.array(u2_cmd) - np.array(u2_hist)
    v2_error = np.array(v2_cmd) - np.array(v2_hist)

    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.suptitle("Adaptive Control of Baloo")
    axs[0].plot(time_hist, u0_error, label="u0_error")
    axs[0].plot(time_hist, v0_error, label="v0_error")
    axs[0].set_title("Joint 0")
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(time_hist, u1_error, label="u1_error")
    axs[1].plot(time_hist, v1_error, label="v1_error")
    axs[1].set_title("Joint 1")
    axs[1].grid()
    axs[1].legend()

    axs[2].plot(time_hist, u2_error, label="u2_error")
    axs[2].plot(time_hist, v2_error, label="v2_error")
    axs[2].set_title("Joint 2")
    axs[2].grid()
    axs[2].legend()

    # fig, axs = plt.subplots(3, 1, sharex=True)
    # fig.suptitle("Adaptive Control of Baloo")
    # axs[0].plot(time_hist, ud0_hist, label="ud0")
    # axs[0].plot(time_hist, ud0_cmd, "--", label="ud0_cmd")
    # axs[0].plot(time_hist, vd0_hist, label="vd0")
    # axs[0].plot(time_hist, vd0_cmd, "--", label="vd0_cmd")
    # axs[0].set_title("Joint 0")
    # axs[0].grid()
    # axs[0].legend()

    # axs[1].plot(time_hist, ud1_hist, label="ud1")
    # axs[1].plot(time_hist, ud1_cmd, "--", label="ud1_cmd")
    # axs[1].plot(time_hist, vd1_hist, label="vd1")
    # axs[1].plot(time_hist, vd1_cmd, "--", label="vd1_cmd")
    # axs[1].set_title("Joint 1")
    # axs[1].grid()
    # axs[1].legend()

    # axs[2].plot(time_hist, ud2_hist, label="ud2")
    # axs[2].plot(time_hist, ud2_cmd, "--", label="ud2_cmd")
    # axs[2].plot(time_hist, vd2_hist, label="vd2")
    # axs[2].plot(time_hist, vd2_cmd, "--", label="vd2_cmd")
    # axs[2].set_title("Joint 2")
    # axs[2].grid()
    # axs[2].legend()

    theta_hatT = np.array(theta_hist).T
    print(len(time_hist))
    print(theta_hatT.shape)  #should be time x M+1 x 6
    fig, axs = plt.subplots(6, 1, sharex=True)
    for i in range(6):
        for j in range(controller.M):
            axs[i].plot(time_hist,
                        theta_hatT[i, j, :],
                        label=f"theta_hatT[{i},{j}]")
        axs[i].set_ylabel(f"theta_{i}")
        axs[i].grid()
        axs[i].legend()
        axs[i].set_yscale('symlog')

    #plot heatmap just to make sure
    plt.figure()
    plt.imshow(theta_hatT[:, :, 100])

    s_hist = np.array(s_hist)
    # print(s_hist.shape)
    fig, axs = plt.subplots(6, 1, sharex=True)
    for i in range(6):
        axs[i].plot(time_hist, s_hist[:, i], label=f"s_{j}")
        axs[i].set_ylabel(f"s_{i}")
        axs[i].grid()

    tau_ff_hist = np.array(tau_ff_hist)
    tau_pd_hist = np.array(tau_pd_hist)
    tau_hist = np.array(tau_hist)
    fig, axs = plt.subplots(6, 1, sharex=True)
    for i in range(6):
        axs[i].plot(time_hist, tau_hist[:, i], label=f"tau_{i}")
        axs[i].plot(time_hist, tau_ff_hist[:, i], label=f"tau_ff_{i}")
        axs[i].plot(time_hist, tau_pd_hist[:, i], label=f"tau_pd_{i}")
        axs[i].set_ylabel(f"tau_{i}")
        axs[i].grid()
        axs[i].legend()

    p_cmds = np.array(pressure_cmd_hist).reshape(-1, 4, 3)
    p_hist = np.array(pressure_hist).reshape(-1, 4, 3)
    pressure_hist = []
    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.suptitle("Pressure Commands")
    for i in range(3):
        axs[i].plot(time_hist, p_cmds[:, 0, i], '--', label="p0")
        axs[i].plot(time_hist, p_cmds[:, 1, i], '--', label="p1")
        axs[i].plot(time_hist, p_cmds[:, 2, i], '--', label="p2")
        axs[i].plot(time_hist, p_cmds[:, 3, i], '--', label="p3")

        axs[i].plot(time_hist, p_hist[:, 0, i], label="p0")
        axs[i].plot(time_hist, p_hist[:, 1, i], label="p1")
        axs[i].plot(time_hist, p_hist[:, 2, i], label="p2")
        axs[i].plot(time_hist, p_hist[:, 3, i], label="p3")
        axs[i].set_title(f"Joint {i}")
        axs[i].grid()
        axs[i].legend()

    plt.show()

    print(f"Average solve time: {np.mean(solve_times)}")