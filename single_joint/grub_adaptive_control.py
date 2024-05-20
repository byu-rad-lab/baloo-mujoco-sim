import rospy
from rad_msgs.msg import PressureStamped
from sensor_msgs.msg import JointState
from control_msgs.msg import JointControllerState
from std_msgs.msg import Float64
import numpy as np
import time
import scipy
from position_adaptive_control import ManipulatorMRACRBF

np.set_printoptions(precision=3, suppress=True)


class GrubAdaptiveController:
    def __init__(self):
        self.create_reference_system()
        Lambda = 150.0  # time constant of sliding surface
        Gamma = 5  # adaptation rate
        K = 1.  # PD gain on s (Kd*Lambda = proportional term in control law, usually a low number 0<Kd<.5)
        RBFmins = (np.array([-10, -10, -10, -10]) * 10.0)
        RBFmaxes = -RBFmins
        numRBFs = 200

        #think a refactor might be good to have each generalized coordiate have its own reference stystem, internal
        # to the controller code. 
        self.mrac = ManipulatorMRACRBF(1, RBFmins, RBFmaxes, numRBFs, Lambda,
                                       Gamma, K)

        rospy.init_node("grub_adaptive")
        self.pub = rospy.Publisher("/robo_0/joint_0/pressure_command",
                                   PressureStamped,
                                   queue_size=1)

        self.pub1 = rospy.Publisher("/robo_0/joint_0/ctrl_info",
                                    JointControllerState,
                                    queue_size=1)

        for i in range(10):
            init = PressureStamped()
            init.header.stamp = rospy.Time.now()
            init.pressure = [0, 0, 0, 0]
            self.pub.publish(init)
            rospy.sleep(0.5)

        self.sub = rospy.Subscriber("/robo_0/joint_0/angle_state",
                                    JointState,
                                    callback=self.callback)

        # self.sub_cmd = rospy.Subscriber("/joint_angle_command", Float64,
        #                                 self.cmd_callback)

        self.ctrl_hist = []
        self.p_hist = []
        self.i_hist = []
        self.d_hist = []
        self.pavg = 100
        self.setpoint = np.array([1.0])

    def create_reference_system(self):

        zeta = 1
        tau = 0.5
        m = 1.0
        b = 2 * m / tau
        k = (b / (2 * zeta))**2

        self.Ades = np.array([
            [-b / m, -k / m],
            [1, 0],
        ])
        self.Bdes = np.array([[k / m], [0]])
        dt = 1 / 500  #rate set by vive jangle estimation
        self.Ad_des = scipy.linalg.expm(self.Ades * dt)
        self.Bd_des = np.matmul(
            np.linalg.inv(self.Ades),
            np.matmul(self.Ad_des - np.eye(self.Ad_des.shape[0]), self.Bdes))
        self.xdes = np.array([0, 0])

    # def cmd_callback(self, msg):
    #     self.setpoint = np.array([msg.position[0]])

    def callback(self, msg):
        rospy.loginfo_once("Starting control loop")
        #forward propogate the reference system
        xdotdes = self.Ades.dot(self.xdes) + self.Bdes.dot(self.setpoint)
        self.xdes = self.Ad_des.dot(self.xdes) + self.Bd_des.dot(self.setpoint)

        q = np.asarray([msg.position[0]])
        qdot = np.asarray([msg.velocity[0]])

        # x = [qdot, q].T
        q_des = np.asarray([self.xdes[1]])
        qd_des = np.asarray([xdotdes[1]])
        qdd_des = np.asarray([xdotdes[0]])

        # print(self.Ades.shape)
        # print(self.Bdes.shape)
        # print(self.Ad_des.shape)
        # print(self.Bd_des.shape)
        # print(self.setpoint.shape)
        # print(xdes.shape)
        # print(xdotdes.shape)
        # print(q.shape)
        # print(qdot.shape)
        # print(q_des.shape)
        # print(qd_des.shape)
        # print(qdd_des.shape)

        ctrl, s, theta_hat = self.mrac.solve_for_next_u(
            q, qdot, q_des, qd_des, qdd_des)
        # print(ctrl)
        # d_custom = self.kd * (msg.velocity[0])
        # ctrl -= d_custom

        p0, p1 = self.torque2pressures(ctrl)

        # print(p0, p1)

        msg = PressureStamped()
        msg.header.stamp = rospy.Time.now()
        msg.pressure = [p0, p1, 0, 0]

        self.pub.publish(msg)

    def torque2pressures(self, torque):
        # convert torque to pressures
        p0 = self.pavg + torque
        p1 = self.pavg - torque

        return p0, p1


if __name__ == "__main__":
    controller = GrubAdaptiveController()
    while not rospy.is_shutdown():
        rospy.spin()
