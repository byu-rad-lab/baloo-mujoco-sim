import rospy
from simple_pid import PID
from rad_msgs.msg import PressureStamped
from sensor_msgs.msg import JointState
import numpy as np
from scipy.signal import butter, lfilter
from collections import deque

np.set_printoptions(precision=3, suppress=True)


class GrubRampController:
    def __init__(self):
        rospy.init_node("grub_pid")
        self.pub = rospy.Publisher("/robo_0/joint_0/pressure_command",
                                   PressureStamped,
                                   queue_size=1)

        self.sub = rospy.Subscriber("/robo_0/joint_0/angle_state",
                                    JointState,
                                    callback=self.callback)

        self.start = rospy.Time.now().to_sec()

    def ramp0(self, t):
        # ramp from 0 to 27.50 in 10 seconds, then back to 0 in 10 seconds
        if t < 10:
            return 27.5 * t
        elif t < 20:
            return 275 - 27.5 * (t - 10)
        else:
            return 0

    def ramp1(self, t):
        # ramp from 0 to 27.50 in 10 seconds, then back to 0 in 10 seconds
        if t < 30:
            return 27.5 * (t - 20)
        elif t < 40:
            return 275 - 27.5 * (t - 30)
        else:
            return 0

    def callback(self, msg):
        msg = PressureStamped()
        msg.header.stamp = rospy.Time.now()
        t = msg.header.stamp.to_sec() - self.start
        if t < 20:
            # ramp chamber 0 up and down in 20 seconds
            ctrl = self.ramp0(t)
            msg.pressure = [ctrl, 0, 0, 0]
        else:
            # then ramp chamber 1 up and down in 20 seconds to bend in the other direction
            ctrl = self.ramp1(t)
            msg.pressure = [0, ctrl, 0, 0]

        # print(msg)
        self.pub.publish(msg)


if __name__ == "__main__":
    controller = GrubRampController()
    while not rospy.is_shutdown():
        rospy.spin()

