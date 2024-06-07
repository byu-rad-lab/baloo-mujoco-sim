from two_link_planar_arm import TwoLinkPlanarArmSlotine
import scipy as sp
import numpy as np
import control as ctrl


class TwoLinkPlanarArmStateSpace:
    def __init__(
        self,
        Q,
        Gamma,
    ) -> None:
        self.arm = TwoLinkPlanarArmSlotine([0., 0.], [0., 0.])
        self.Theta_hat = np.zeros(4)

    def calc_AB(self, q, qd):
        M = self.arm.calc_H(q, self.arm.a)
        C = self.arm.calc_C(q, qd, self.arm.a)

        Minv = np.linalg.inv(M)

        A = np.bmat([[-Minv @ C, np.zeros((2, 2))],
                     [np.eye(2), np.zeros((2, 2))]])

        B = np.bmat([[Minv], [np.zeros((2, 2))]])

        return np.asarray(A), np.asarray(B)

    def Phi(self, x):
        pass

    def calc_u(self, r):
        u = self.Kx_hat.T @ self.x + self.Kr_hat.T @ r


if __name__ == '__main__':
    Q = np.eye(4)
    Gamma = np.eye(2)
    two_link_planar_arm_statespace = TwoLinkPlanarArmStateSpace(Q, Gamma)
    print(*two_link_planar_arm_statespace.calc_AB([0., 0.], [0., 0.]))

    C = ctrl.ctrb(*two_link_planar_arm_statespace.calc_AB([0., 0.], [0., 0.]))

    print(np.linalg.matrix_rank(C))
