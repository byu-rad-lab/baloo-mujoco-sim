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
        numberOfRBFCenters: int,
        RBFmins: npt.NDArray,
        RBFmaxes: npt.NDArray,
        zeta=npt.NDArray,
        time_constant=npt.NDArray,
        Lambda=1.0,
        Gamma=1.0,
        KD=1.0,
        ctrl_dt=0.0,
    ):
        assert len(
            RBFmins) == 4 * num_gen_coords  #4 bc of the [q,qd,qd_r,qdd_r]
        assert len(RBFmaxes) == 4 * num_gen_coords
        assert len(zeta) == num_gen_coords
        assert len(time_constant) == num_gen_coords

        self.numberGenCoords = num_gen_coords
        self.N = numberOfRBFCenters
        self.RBFmins = RBFmins  # should be same length as x in Phi(x)
        self.RBFmaxes = RBFmaxes  # should be same length as x in Phi(x)

        self.theta_hat = np.zeros([numberOfRBFCenters + 1, num_gen_coords])
        self.Gamma = np.eye(numberOfRBFCenters + 1) * Gamma
        self.Lambda = np.eye(num_gen_coords) * Lambda
        self.Kd = np.eye(num_gen_coords) * KD
        self.dt = ctrl_dt

        self._create_desired_system(zeta, time_constant, ctrl_dt)

    def _create_desired_system(self, zeta, tau, dt):
        # A_ref (2*num_gen_coords x 2*num_gen_coords)
        # B_ref (2*num_gen_coords x num_gen_forces)

        #make num_gen_coords parameters, from second order system dynamics
        m = 1.0
        b = 2 * m / tau
        k = (b / (2 * zeta))**2

        self.x_des = np.zeros([2 * self.numberGenCoords])
        self.xdot_des = np.zeros([2 * self.numberGenCoords])

        self.Ades_stack = np.zeros(
            [2 * self.numberGenCoords, 2 * self.numberGenCoords])
        self.Bdes_stack = np.zeros(
            [2 * self.numberGenCoords, self.numberGenCoords])
        self.Ad_des_stack = np.zeros(
            [2 * self.numberGenCoords, 2 * self.numberGenCoords])
        self.Bd_des_stack = np.zeros(
            [2 * self.numberGenCoords, self.numberGenCoords])

        #create Aref, Bref, Ad_ref, Bd_ref for each generalized coordiate
        for i in range(self.numberGenCoords):
            Ades = np.array([
                [-b[i] / m, -k[i] / m],
                [1, 0],
            ])
            Bdes = np.array([[k[i] / m], [0]])

            #discrete time, see https://en.wikipedia.org/wiki/Discretization#Derivation
            Ad_des = scipy.linalg.expm(Ades * dt)
            Bd_des = np.matmul(
                np.linalg.inv(Ades),
                np.matmul(Ad_des - np.eye(Ad_des.shape[0]), Bdes))

            self.Ades_stack[i * 2:i * 2 + 2, i * 2:i * 2 + 2] = Ades
            self.Bdes_stack[i * 2:i * 2 + 2, i] = Bdes.flatten()
            self.Ad_des_stack[i * 2:i * 2 + 2, i * 2:i * 2 + 2] = Ad_des
            self.Bd_des_stack[i * 2:i * 2 + 2, i] = Bd_des.flatten()

    def _update_desired_trajectory(self, u: npt.NDArray):
        # x_des (2*num_gen_coords x 1)
        # r (num_gen_coords x 1)
        #xdot_des = [qdd, qd] * num_gen_coords
        #x_des = [qd, q] * num_gen_coords
        self.xdot_des = self.Ades_stack @ self.x_des + self.Bdes_stack @ u
        self.x_des = self.Ad_des_stack @ self.x_des + self.Bd_des_stack @ u

        qddot_des = self.xdot_des[::2]
        qdot_des = self.xdot_des[1::2]
        q_des = self.x_des[1::2]

        return q_des, qdot_des, qddot_des

    def _calc_regressor(
        self,
        q: npt.NDArray,
        qdot: npt.NDArray,
        qdot_ref: npt.NDArray,
        qddot_ref: npt.NDArray,
    ):
        assert len(q) == self.numberGenCoords
        assert len(qdot) == self.numberGenCoords
        assert len(qdot_ref) == self.numberGenCoords
        assert len(qddot_ref) == self.numberGenCoords

        centers = np.linspace(self.RBFmins, self.RBFmaxes, self.N)
        dMax = np.linalg.norm(centers[1, :] - centers[0, :])
        x = np.hstack([q, qdot, qdot_ref, qddot_ref])
        norms = np.linalg.norm(x - centers, axis=1)
        width = self.N / dMax**2
        Phi = np.exp(-width * norms**2)

        return np.append(Phi, 1.0)

    def _calc_refs(
        self,
        q: npt.NDArray[np.float64],
        qdot: npt.NDArray[np.float64],
        q_des: npt.NDArray[np.float64],
        qdot_des: npt.NDArray[np.float64],
        qddot_des: npt.NDArray[np.float64],
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

        qtilde_dot = qdot - qdot_des
        qddot_ref = qddot_des - self.Lambda.dot(qtilde_dot)

        qtilde = q - q_des
        qdot_ref = qdot_des - self.Lambda.dot(qtilde)
        return qdot_ref, qddot_ref

    def _update_weights(self, s: npt.NDArray, Phi: npt.NDArray) -> None:
        thetaDot = -self.Gamma @ np.outer(Phi, s)
        self.theta_hat = self.theta_hat + thetaDot * self.dt

    def solve_for_next_u(
        self,
        q: npt.NDArray[np.float64],
        qdot: npt.NDArray[np.float64],
        q_des: npt.NDArray[np.float64],
        qdot_des: npt.NDArray[np.float64] = None,
        qddot_des: npt.NDArray[np.float64] = None,
        adapt=True,
    ) -> npt.NDArray[np.float64]:

        #given qd_des and qdd_des from somewhere, I calculate a desired trajectory. A conventient
        # choice for obtaining qd_des and qdd_des is to use a critically damped 2nd roder system.
        #if qdot_des and qddot_des are not provided, I will use the internal trajectory generator to calculate them.

        if qdot_des is None or qddot_des is None:
            q_des, qdot_des, qddot_des = self._update_desired_trajectory(q_des)

        qdot_ref, qddot_ref = self._calc_refs(q, qdot, q_des, qdot_des,
                                              qddot_des)
        Phi = self._calc_regressor(q, qdot, qdot_ref, qddot_ref)
        s = q - qdot_ref

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

    # Load model for simulation
    model = mujoco.MjModel.from_xml_path(
        "/home/curtis/baloo_mujoco_sim/single_joint/single_joint.xml")
    data = mujoco.MjData(model)

    Lambda = 75.0  # time constant of sliding surface
    Gamma = 5  # adaptation rate
    K = 0.1  # PD gain on s (Kd*Lambda = proportional term in control law, usually a low number 0<Kd<.5)
    RBFmins = np.array([-100] * 8) * 1.0
    RBFmaxes = -RBFmins
    numRBFs = 200
    controller = ManipulatorMRACRBF(
        2,
        numRBFs,
        RBFmins,
        RBFmaxes,
        np.array([1, 1]),
        np.array([.2, .2]),
        Lambda,
        Gamma,
        K,
        model.opt.timestep,
    )

    plotter = JointAnglePlotter(30, model.opt.timestep)
    plotter.show()

    paused = False

    def torque2pressures(torque):
        # convert torque to pressures
        p0 = 200 + torque
        p1 = 200 - torque

        return p0, p1

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

                q = np.asarray(data.sensor("left_0").data[:2])
                qdot = np.asarray(data.sensor("left_0").data[2:])

                q_des = np.asarray([1.111, -1.111])

                # start = time.time()
                tau, s, theta_hat = controller.solve_for_next_u(q, qdot, q_des)
                print(s)

                ctrl = tau / 2 * 50
                p0, p1 = torque2pressures(ctrl[0])
                p2, p3 = torque2pressures(ctrl[1])

                data.ctrl = np.array([p0, p1, p2, p3])
                # print(ctrl)

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(model, data)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()
                plotter.update(
                    model,
                    data,
                    {
                        "u_cmd": q_des[0],
                        # "s": s.item(),
                        # "theta_hat": theta_hat
                    })

                # Rudimentary time keeping, will drift relative to wall clock.
                # print(time.time() - step_start)
                time_until_next_step = model.opt.timestep - (time.time() -
                                                             step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    plotter.close()
