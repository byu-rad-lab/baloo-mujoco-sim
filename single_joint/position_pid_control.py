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

    pid = PID(200, 80, 1, setpoint=.5)
    pid2 = PID(200, 80, 1, setpoint=.5)
    pid.output_limits = (-200, 200)
    pid2.output_limits = (-200, 200)
    pid.sample_time = model.opt.timestep
    pid2.sample_time = model.opt.timestep

    plotter = JointAnglePlotter(30, model.opt.timestep)
    plotter.show()

    def torque2pressures(torque):
        # convert torque to pressures
        p0 = 200 + torque
        p1 = 200 - torque

        return p0, p1

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()

        # with viewer.lock():q
        #     #update gui things
        #     mujoco.mjr_figure(viewport, fig, context)

        while viewer.is_running():
            step_start = time.time()

            ctrl_u = pid(data.sensor("left_0").data[0])
            ctrl_v = pid2(data.sensor("left_0").data[1])

            # print(ctrl_u, ctrl_v)

            p0, p1 = torque2pressures(ctrl_u)
            p2, p3 = torque2pressures(ctrl_v)

            print(p0, p1, p2, p3)
            data.ctrl[0] = p0
            data.ctrl[1] = p1
            data.ctrl[2] = p2
            data.ctrl[3] = p3

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(model, data)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            plotter.update(model, data, {"u_cmd": pid.setpoint})

            # Rudimentary time keeping, will drift relative to wall clock.
            # print(time.time() - step_start)
            time_until_next_step = model.opt.timestep - (time.time() -
                                                         step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    plotter.close()
