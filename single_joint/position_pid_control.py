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

    pid = PID(120, 80, 3, setpoint=.5)
    pid.output_limits = (0, 400)
    pid.sample_time = model.opt.timestep

    plotter = JointAnglePlotter(30, model.opt.timestep)
    plotter.show()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()

        # with viewer.lock():q
        #     #update gui things
        #     mujoco.mjr_figure(viewport, fig, context)

        while viewer.is_running():
            step_start = time.time()

            ctrl = pid(data.sensor("left_0").data[0])
            print(ctrl)
            data.ctrl[0] = ctrl
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(model, data)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            plotter.update(model, data, {"u_cmd": pid.setpoint})

            # Rudimentary time keeping, will drift relative to wall clock.
            # print(time.time() - step_start)
            # time_until_next_step = model.opt.timestep - (time.time() -
            #  step_start)
            # if time_until_next_step > 0:
            # time.sleep(time_until_next_step)

    plotter.close()
