import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm


class TwoLinkPlanarArmSlotine:
    def __init__(self, q_init, qdot_init):
        self.ml = 1.0
        self.l1 = 1
        self.me = 2
        self.delta_e = 30
        self.Il = .12
        self.Icl = 0.5
        self.Ie = .25
        self.lce = .6

        self.a1 = self.Il + self.ml * self.Icl**2 + self.Ie + self.me * self.lce**2 + self.me * self.l1**2
        self.a2 = self.Ie + self.me * self.lce**2
        self.a3 = self.me * self.l1 * self.lce * np.cos(
            np.deg2rad(self.delta_e))
        self.a4 = self.me * self.l1 * self.lce * np.sin(
            np.deg2rad(self.delta_e))

        self.a = [self.a1, self.a2, self.a3, self.a4]

        # print(self.a1, self.a2, self.a3, self.a4)

        # create a figure and lines
        self.fig, self.ax = plt.subplots()
        self.line1, = self.ax.plot([], [], 'r-')  # link 1
        self.line2, = self.ax.plot([], [], 'b-')  # link 2
        self.joint1, = self.ax.plot([], [], 'ko')  # joint 1
        self.joint2, = self.ax.plot([], [], 'ko')  # joint 2

        # set the limits and aspect ratio
        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
        self.ax.set_aspect('equal', adjustable='box')

    def calc_H(self, q, a):
        q1, q2 = q
        a1, a2, a3, a4 = a

        H11 = a1 + 2 * a3 * np.cos(q2) + 2 * a4 * np.sin(q2)
        H12 = H21 = a2 + a3 * np.cos(q2) + a4 * np.sin(q2)
        H22 = a2

        H = np.array([[H11, H12], [H21, H22]])

        return H

    def calc_C(self, q, qdot, a):
        q1, q2 = q
        q1dot, q2dot = qdot
        a1, a2, a3, a4 = a

        h = a3 * np.sin(q2) - a4 * np.cos(q2)
        C11 = -h * q2dot
        C12 = -h * (q1dot + q2dot)
        C21 = h * q1dot
        C22 = 0

        C = np.array([[C11, C12], [C21, C22]])

        return C

    def simulate_dynamics(self, q, qd, tau, dt):
        H = self.calc_H(q, self.a)
        C = self.calc_C(q, qd, self.a)

        qdd = np.linalg.inv(H) @ (tau - C @ qd)

        qd += qdd * dt
        q += qd * dt

        return q, qd

    def visualize(self, q):
        q1, q2 = q
        # check if the figure window is still open
        if not plt.fignum_exists(self.fig.number):
            return
        # calculate the end-points of the two links
        x1 = self.l1 * np.cos(q1)
        y1 = self.l1 * np.sin(q1)
        x2 = x1 + self.lce * np.cos(q1 + q2)
        y2 = y1 + self.lce * np.sin(q1 + q2)

        # update the lines and joints
        self.line1.set_data([0, x1], [0, y1])
        self.line2.set_data([x1, x2], [y1, y2])
        self.joint1.set_data([0, x1], [0, y1])
        self.joint2.set_data([x1, x2], [y1, y2])

        # update the limits
        self.ax.set_xlim([-self.a1 - self.a2, self.a1 + self.a2])
        self.ax.set_ylim([-self.a1 - self.a2, self.a1 + self.a2])
        self.ax.set_aspect('equal', adjustable='box')

        # redraw the canvas
        self.fig.canvas.draw()
        plt.pause(0.00001)


class AdaptiveController:
    #implementation of adaptive controller in section 8.5.4 of siciliano
    def __init__(self):
        self.pi_hat = np.zeros(8)
        self.Kd = np.eye(2) * 100
        self.Kpi = np.eye(8)
        self.Lambda = np.eye(2) * 20
        self.a_hat = np.zeros(4)
        self.dt = 0.001
        self.Gamma = np.diag([.03, .05, .1, .3])

        self.arm = TwoLinkPlanarArmSlotine([0., 0.], [0., 0.])

    def upate_adaptive_params(self, Y, s):

        ahat_dot = -self.Gamma @ Y.T @ s
        self.a_hat += ahat_dot * self.dt

    def calc_Y_regressor(self, q, qd, qd_ref, qdd_ref):
        q1, q2 = q
        q1d, q2d = qd
        q1d_ref, q2d_ref = qd_ref
        q1dd_ref, q2dd_ref = qdd_ref

        Y11 = q1dd_ref
        Y12 = q2dd_ref
        Y13 = ((2 * q1dd_ref + q2dd_ref) * np.cos(q2) -
               (q2d * q1d_ref + q1d * q2d_ref + q2d * q2d_ref) * np.sin(q2))

        Y14 = ((2 * q1dd_ref + q2dd_ref) * np.sin(q2) +
               (q2d * q1d_ref + q1d * q2d_ref + q2d * q2d_ref) * np.cos(q2))

        Y21 = 0
        Y22 = q1dd_ref + q2dd_ref
        Y23 = q1dd_ref * np.cos(q2) + q1d * q1d_ref * np.sin(q2)
        Y24 = q1dd_ref * np.sin(q2) - q1d * q1d_ref * np.cos(q2)

        Y = np.array([[Y11, Y12, Y13, Y14], [Y21, Y22, Y23, Y24]])

        return Y

    def calc_torques(self, q, qd, q_des, qd_des, qdd_des):

        qd_ref = qd_des - self.Lambda @ (q - q_des)
        qdd_ref = qdd_des - self.Lambda @ (qd - qd_des)

        H = self.arm.calc_H(q, self.a_hat)
        C = self.arm.calc_C(q, qd, self.a_hat)

        s = qd - qd_ref

        tau = H @ qdd_ref + C @ qd_ref - self.Kd @ s

        Y = self.calc_Y_regressor(q, qd, qd_ref, qdd_ref)

        self.upate_adaptive_params(Y, s)

        return tau, self.a_hat, s


if __name__ == "__main__":
    # initial state
    q_init = [0., 0.]
    qdot_init = [0., 0.]

    # create a two link planar arm object
    # two_link_planar_arm = TwoLinkPlanarArmSlotine(q_init, qdot_init)
    controller = AdaptiveController()

    def calculate_desired_trajectory(t):
        q1_des = np.deg2rad(30) * (1 - np.cos(2 * np.pi * t))
        q2_des = np.deg2rad(45) * (1 - np.cos(2 * np.pi * t))

        qd1_des = np.deg2rad(30) * 2 * np.pi * np.sin(2 * np.pi * t)
        qd2_des = np.deg2rad(45) * 2 * np.pi * np.sin(2 * np.pi * t)

        qdd1_des = np.deg2rad(30) * (2 * np.pi)**2 * np.cos(2 * np.pi * t)
        qdd2_des = np.deg2rad(45) * (2 * np.pi)**2 * np.cos(2 * np.pi * t)

        return [q1_des, q2_des], [qd1_des, qd2_des], [qdd1_des, qdd2_des]

    # simulate the dynamics
    dt = 0.001 #needs to be this small for numerical stability
    q_values = []  # list to store q values
    a_hist = []
    s_hist = []
    tau_hist = []
    qerr_hist = []
    t_hist = []
    q = np.array([0., 0.])
    qd = np.array([0., 0.])
    seconds = 60 * 30
    for i in tqdm(range(int(seconds / dt))):
        t = i * dt
        q_des, qd_des, qdd_des = calculate_desired_trajectory(t)
        tau, a_hat, s = controller.calc_torques(q, qd, q_des, qd_des, qdd_des)
        q, qd = controller.arm.simulate_dynamics(q, qd, tau, dt)
        # controller.arm.visualize(q)
        t_hist.append(t)
        q_values.append(deepcopy(q))  # store the current q value
        a_hist.append(deepcopy(a_hat))
        s_hist.append(deepcopy(s))
        tau_hist.append(deepcopy(tau))
        qerr_hist.append(deepcopy(q - q_des))

    # plot q over time
    t_hist = np.array(t_hist)[::10]
    q_values = np.array(q_values)[::10]
    plt.figure()
    plt.plot(t_hist, q_values[:, 0], label='vartheta1')
    plt.plot(t_hist, q_values[:, 1], label='vartheta2')
    plt.xlabel('Time step')
    plt.ylabel('Configuration')
    plt.legend()

    a_hist = np.array(a_hist)[::10]
    plt.figure()
    plt.plot(t_hist, a_hist[:, 0], 'r', label='a1_hat')
    plt.plot(t_hist, a_hist[:, 1], 'g', label='a2_hat')
    plt.plot(t_hist, a_hist[:, 2], 'k', label='a3_hat')
    plt.plot(t_hist, a_hist[:, 3], 'b', label='a4_hat')
    plt.plot(t_hist, [controller.arm.a1] * len(t_hist), 'r--', label='a1')
    plt.plot(t_hist, [controller.arm.a2] * len(t_hist), 'g--', label='a2')
    plt.plot(t_hist, [controller.arm.a3] * len(t_hist), 'k--', label='a3')
    plt.plot(t_hist, [controller.arm.a4] * len(t_hist), 'b--', label='a4')
    plt.xlabel('Time step')
    plt.ylabel('Adaptive parameters')
    plt.legend()

    s_hist = np.array(s_hist)[::10]
    plt.figure()
    plt.plot(t_hist, s_hist[:, 0], label='s1')
    plt.plot(t_hist, s_hist[:, 1], label='s2')
    plt.xlabel('Time step')
    plt.ylabel('s')
    plt.legend()

    tau_hist = np.array(tau_hist)[::10]
    plt.figure()
    plt.plot(t_hist, tau_hist[:, 0], label='tau1')
    plt.plot(t_hist, tau_hist[:, 1], label='tau2')
    plt.xlabel('Time step')
    plt.ylabel('Control input')
    plt.legend()

    qerr_hist = np.array(qerr_hist)[::10]
    plt.figure()
    plt.plot(t_hist, np.rad2deg(qerr_hist[:, 0]), label='q1 error')
    plt.plot(t_hist, np.rad2deg(qerr_hist[:, 1]), label='q2 error')
    plt.xlabel('Time step')
    plt.ylabel('Error')
    plt.legend()

    plt.show()
