import mujoco
from single_joint.position_adaptive_control import ManipulatorMRACRBF
import numpy as np
import mujoco.viewer as viewer
import time
from utils.baloo_mj_api import get_joint_angles, get_joint_vel, set_joint_pressure_commands
from utils.mjData_plotter import MjDataPlotter

np.set_printoptions(precision=3, suppress=True)

#load baloo.xml file
model = mujoco.MjModel.from_xml_path("baloo.xml")
data = mujoco.MjData(model)

controller = ManipulatorMRACRBF(
    num_gen_coords=6,
    numberOfRBFCenters=
    20,  #number doesn't seem to matter too much for performance, but more smooths out the control signal
    RBFmins=np.array([-200] * 6 * 4) * 1.0,
    RBFmaxes=np.array([200] * 6 * 4) * 1.0,
    zeta=np.array([1] * 6),
    time_constant=np.array([.2] * 6),
    Lambda=2,  #weight on position error qdes - q
    Gamma=150,  #don't love having this so high, its kind of like the itegrator
    KD=
    150,  #weight on s, (directly velocity error, KD*lambda for equivalent position error)
    ctrl_dt=model.opt.timestep,
)

plotter = MjDataPlotter(30, model.opt.timestep)

first_time = True

with viewer.launch_passive(model, data) as viewer:
    start = time.time()

    #disable pressure control sliders since user can't control them here.

    while viewer.is_running():
        if first_time:
            input("Press Enter to continue...")
            first_time = False
        step_start = time.time()

        q0 = get_joint_angles(model, data, "left", 0)
        q1 = get_joint_angles(model, data, "left", 1)
        q2 = get_joint_angles(model, data, "left", 2)

        q0dot = get_joint_vel(model, data, "left", 0)
        q1dot = get_joint_vel(model, data, "left", 1)
        q2dot = get_joint_vel(model, data, "left", 2)

        q = np.hstack([q0, q1, q2]).squeeze()
        qdot = np.hstack([q0dot, q1dot, q2dot]).squeeze()

        q_des = np.asarray([0.7, -0.5, -0.25, 0.4, -0.7, 0.5])

        # start = time.time()
        tau, s, theta_hat = controller.solve_for_next_u(q, qdot, q_des)

        # print(q_des - q)
        # print(s)
        # tau = tau * 1e3
        print(f"\n\ntau: {tau}")
        print(f"s: {s}")

        for i in range(3):
            p0, p1 = controller.torque2pressures(tau[i * 2])
            p2, p3 = controller.torque2pressures(tau[(2 * i) + 1])

            set_joint_pressure_commands(model, data, "left", i,
                                        [p0, p1, p2, p3])

        # data.ctrl = np.array([p0, p1, p2, p3])
        # print(theta_hat.shape)

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(model, data)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # plotter.update(model, data, {
        #     "joint0_angle0_cmd": q_des[0],
        #     "joint0_angle1_cmd": q_des[1],
        #     "joint1_angle0_cmd": q_des[2],
        #     "joint1_angle1_cmd": q_des[3],
        #     "joint2_angle0_cmd": q_des[4],
        #     "joint2_angle1_cmd": q_des[5],
        # })

        # Rudimentary time keeping, will drift relative to wall clock.
        # print(time.time() - step_start)
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
