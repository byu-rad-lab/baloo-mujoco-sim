import numpy as np
import numpy.typing as npt
from typing import Callable, Tuple
from tqdm import tqdm
import mujoco
import mujoco.viewer as viewer
import time

import scipy
from copy import deepcopy
import matplotlib.pyplot as plt
from plotter import JointAnglePlotter


class ManipulatorMRACRBF:
    def __init__(
        self,
        num_gen_coords: int,
        RBFmins: npt.NDArray,
        RBFmaxes: npt.NDArray,
        numberOfRBFCenters: int,
        Lambda=1.0,
        Gamma=1.0,
        KD=1.0,
    ):
        self.numberGenCoords = num_gen_coords
        self.N = numberOfRBFCenters
        self.RBFmins = RBFmins  # should be same length as x in Phi(x)
        self.RBFmaxes = RBFmaxes  # should be same length as x in Phi(x)

        assert len(RBFmins) == 4 * num_gen_coords
        assert len(RBFmaxes) == 4 * num_gen_coords

        self.theta_hat = np.zeros([numberOfRBFCenters + 1, num_gen_coords])
        self.Gamma = np.eye(numberOfRBFCenters + 1) * Gamma
        self.Lambda = np.eye(num_gen_coords) * Lambda
        self.Kd = np.eye(num_gen_coords) * KD

        # self.Lambda = np.diag([Lambda, Lambda, 3., 3., 2.0348971137496514, 2.0348971137496514])
        # self.Kd = np.diag([KD, KD, 1.5, 1.5, 4.035313436917082, 4.035313436917082])

        self.dt = 0.001

    def _calc_regressor(
        self,
        q: npt.NDArray,
        qdot: npt.NDArray,
        qdot_r: npt.NDArray,
        qddot_r: npt.NDArray,
    ):
        assert len(q) == self.numberGenCoords
        assert len(qdot) == self.numberGenCoords
        assert len(qdot_r) == self.numberGenCoords
        assert len(qddot_r) == self.numberGenCoords

        centers = np.linspace(self.RBFmins, self.RBFmaxes, self.N)
        dMax = np.linalg.norm(centers[1, :] - centers[0, :])
        x = np.hstack([q, qdot, qdot_r, qddot_r])
        norms = np.linalg.norm(x - centers, axis=1)
        width = self.N / dMax**2
        Phi = np.exp(-width * norms**2)

        return np.append(Phi, 1.0)

    def _calc_s(self, qdot: npt.NDArray[np.float64],
                qdot_r: npt.NDArray[np.float64]):
        """
        Sliding surface corresponding to GES tracking error dynamics
        between q and q_des.
        """
        s = qdot - qdot_r
        return s

    def _calc_refs(
        self,
        q: npt.NDArray[np.float64],
        qdot: npt.NDArray[np.float64],
        q_d: npt.NDArray[np.float64],
        qdot_d: npt.NDArray[np.float64],
        qddot_d: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Equations 7b,7c in Slotine paper. Note that q_r is not
        explicitly needed in control law, so it is not calculated
        here.


        Args:
            q (npt.NDArray[np.float64]): joint angles
            qdot (npt.NDArray[np.float64]): joint vel
            q_d (npt.NDArray[np.float64]): desired joint angles
            qdot_d (npt.NDArray[np.float64]): desired joint vel
            qddot_d (npt.NDArray[np.float64]): desired joint accel

        Returns:
            tuple: (reference velocity, reference acceleration)
        """

        qddot_r = qddot_d - self.Lambda.dot(qdot - qdot_d)

        qdot_r = qdot_d - self.Lambda.dot(q - q_d)
        return qdot_r, qddot_r

    def _update_weights(self, s: npt.NDArray, Phi: npt.NDArray) -> None:
        thetaDot = -self.Gamma @ np.outer(Phi, s)
        self.theta_hat = self.theta_hat + thetaDot * self.dt

    def solve_for_next_u(
        self,
        q: npt.NDArray[np.float64],
        qdot: npt.NDArray[np.float64],
        q_d: npt.NDArray[np.float64],
        qdot_d: npt.NDArray[np.float64],
        qddot_d: npt.NDArray[np.float64],
        adapt=True,
    ) -> npt.NDArray[np.float64]:
        """
        External API function that should be called to calculate control input.

        Args:
            q (npt.NDArray[np.float64]): _description_
            qdot (npt.NDArray[np.float64]): _description_
            q_d (npt.NDArray[np.float64]): _description_
            qdot_d (npt.NDArray[np.float64]): _description_
            qddot_d (npt.NDArray[np.float64]): _description_
            adapt (bool, optional): If true, weights will update in response to tracking error. Defaults to True.

        Returns:
            npt.NDArray[np.float64]: joint torques to apply to robot
        """
        qdot_r, qddot_r = self._calc_refs(q, qdot, q_d, qdot_d, qddot_d)
        s = self._calc_s(qdot, qdot_r)
        Phi = self._calc_regressor(q, qdot, qdot_r, qddot_r)

        if adapt:
            self._update_weights(s, Phi)

        tau = self.theta_hat.T @ Phi - self.Kd @ s

        return tau, s, self.theta_hat


if __name__ == "__main__":

    def u_to_pressures(u, pmean):
        pressures = np.matrix([
            [pmean + u[0] / 2 * 100.0],
            [pmean - u[0] / 2 * 100.0],
            [pmean + u[1] / 2 * 100.0],
            [pmean - u[1] / 2 * 100.0],
        ])
        return pressures

    Lambda = 150.0  # time constant of sliding surface
    Gamma = 10  # adaptation rate
    K = 0.05  # PD gain on s (Kd*Lambda = proportional term in control law, usually a low number 0<Kd<.5)
    RBFmins = (np.array([-np.pi, -np.pi, -np.pi, -np.pi]) * 1.0)
    RBFmaxes = -RBFmins
    numRBFs = 20
    controller = ManipulatorMRACRBF(1, RBFmins, RBFmaxes, numRBFs, Lambda,
                                    Gamma, K)
    # Load model for simulation
    model = mujoco.MjModel.from_xml_path(
        "/home/curtis/baloo_mujoco_sim/single_joint/single_joint.xml")
    data = mujoco.MjData(model)

    # Create a reference system to track, critically damped, dc gain of 1 (from k/m in B), tau of .2, x= [qdot,q].T
    dt = model.opt.timestep
    m = 2.0
    b = 20.0
    k = 50.0
    Ades = np.array([
        [-b / m, -k / m],
        [1, 0],
    ])
    Bdes = np.array([[k / m], [0]])
    Ad_des = scipy.linalg.expm(Ades * dt)
    Bd_des = np.matmul(np.linalg.inv(Ades),
                       np.matmul(Ad_des - np.eye(Ad_des.shape[0]), Bdes))
    xdes = np.array([0, 0])

    r = np.array([0.5])

    plotter = JointAnglePlotter(30, model.opt.timestep)
    plotter.show()

    paused = False

    def key_callback(keycode):
        if keycode == ord(' '):
            global paused
            paused = not paused

    with mujoco.viewer.launch_passive(model, data,
                                      key_callback=key_callback) as viewer:
        start = time.time()

        #disable pressure control sliders since user can't control them here.

        while viewer.is_running():
            if not paused:
                step_start = time.time()

                xdotdes = Ades.dot(xdes) + Bdes.dot(r)
                xdes = Ad_des.dot(xdes) + Bd_des.dot(r)

                q = np.asarray([data.sensor("left_0").data[0]])
                qdot = np.asarray([data.sensor("left_0").data[2]])

                q_des = np.asarray([xdes[1]])
                qd_des = np.asarray([xdotdes[1]])
                qdd_des = np.asarray([xdotdes[0]])

                # start = time.time()
                tau, s, theta_hat = controller.solve_for_next_u(
                    q, qdot, q_des, qd_des, qdd_des)
                # print(time.time() - start)

                ctrl = tau / 2 * 50
                data.ctrl[0] = ctrl
                # print(ctrl)

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(model, data)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()
                plotter.update(model, data, {
                    "u_cmd": r[0],
                    "s": s.item(),
                    "theta_hat": theta_hat
                })

                # Rudimentary time keeping, will drift relative to wall clock.
                # print(time.time() - step_start)
                time_until_next_step = model.opt.timestep - (time.time() -
                                                             step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    plotter.close()
