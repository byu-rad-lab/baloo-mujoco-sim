import mujoco
import mujoco.viewer as viewer
import time
from plotter import JointAnglePlotter
from simple_pid import PID

if __name__ == "__main__":
    f = open("single_joint.xml", "r")

    # Load model for simulation.
    model = mujoco.MjModel.from_xml_path(f.name)
    data = mujoco.MjData(model)

    pid = PID(200, 80, 1, setpoint=0)
    pid2 = PID(200, 80, 1, setpoint=0)
    pid.output_limits = (-200, 200)
    pid2.output_limits = (-200, 200)
    pid.sample_time = model.opt.timestep
    pid2.sample_time = model.opt.timestep

    # plotter = JointAnglePlotter(30, model.opt.timestep)
    # plotter.show()


    def torque2pressures(torque):
        # convert torque to pressures
        p0 = 200 + torque
        p1 = 200 - torque

        return p0, p1

    prev_setpoint1 = 0
    prev_setpoint2 = 0

    def update_setpoint(t, controller1, controller2):
        alpha = .01  # replace with your actual alpha value
        global prev_setpoint1, prev_setpoint2

        if t < 10:
            desired_setpoint1 = 0.2
            desired_setpoint2 = -1.1
        elif t < 30:
            desired_setpoint1 = 0.0
            desired_setpoint2 = 0.0
        elif t < 40:
            desired_setpoint1 = -0.7
            desired_setpoint2 = 0.9
        else:
            desired_setpoint1 = 0
            desired_setpoint2 = 0

        # Apply the prefilter
        controller1.setpoint = alpha * desired_setpoint1 + (
            1 - alpha) * prev_setpoint1
        controller2.setpoint = alpha * desired_setpoint2 + (
            1 - alpha) * prev_setpoint2

        # Update the previous setpoint
        prev_setpoint1 = controller1.setpoint
        prev_setpoint2 = controller2.setpoint

    first_time = True

    time_hist = []
    u_history = []
    v_history = []
    ucmd_history = []
    vcmd_history = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()

        # with viewer.lock():q
        #     #update gui things
        #     mujoco.mjr_figure(viewport, fig, context)

        while viewer.is_running():
            if first_time:
                input("Press Enter to continue...")
                first_time = False
            step_start = time.time()

            update_setpoint(data.time, pid, pid2)
            # print(pid.setpoint, pid2.setpoint)
            ctrl_u = pid(data.sensor("left_0").data[0])
            ctrl_v = pid2(data.sensor("left_0").data[1])

            # print(ctrl_u, ctrl_v)

            p0, p1 = torque2pressures(ctrl_u)
            p2, p3 = torque2pressures(ctrl_v)

            # print(p0, p1, p2, p3)
            data.ctrl[0] = p0
            data.ctrl[1] = p1
            data.ctrl[2] = p2
            data.ctrl[3] = p3

            time_hist.append(data.time)
            u_history.append(data.sensor("left_0").data[0])
            v_history.append(data.sensor("left_0").data[1])
            ucmd_history.append(pid.setpoint)
            vcmd_history.append(pid2.setpoint)

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(model, data)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # plotter.update(model, data, {"u_cmd": pid.setpoint})

            # Rudimentary time keeping, will drift relative to wall clock.
            # print(time.time() - step_start)
            time_until_next_step = model.opt.timestep - (time.time() -
                                                         step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            # print(data.time)
            if data.time > 50:
                break

    # plotter.close()

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(time_hist, u_history, label="u")
    plt.plot(time_hist, v_history, label="v")
    plt.plot(time_hist, ucmd_history, '--', label="u_cmd")
    plt.plot(time_hist, vcmd_history, '--', label="v_cmd")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.grid()
    plt.show()
