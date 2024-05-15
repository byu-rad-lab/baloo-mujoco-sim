import rospy
from simple_pid import PID
from rad_msgs.msg import PressureStamped
from sensor_msgs.msg import JointState
from control_msgs.msg import JointControllerState
from std_msgs.msg import Float64
import numpy as np
from scipy.signal import butter, lfilter
from collections import deque
import time

np.set_printoptions(precision=3, suppress=True)


class GrubPIDController:
    def __init__(self):
        self.pid = PID(
            170, 25, 3, setpoint=.5
        )  # tuned somewhat conservatively, since that's the best PID can do RN.
        self.pid2 = PID(170, 25, 3, setpoint=-.5)
        self.kd = 0
        self.pid.output_limits = (-200, 200)
        self.pid2.output_limits = (-200, 200)
        self.pid.sample_time = 1 / 650
        self.pid2.sample_time = 1 / 650
        self.prev_position = 0
        self.prev_position2 = 0

        self.prev_positions = deque(
            maxlen=4)  # Buffer for the previous positions
        self.prev_positions2 = deque(
            maxlen=4)  # Buffer for the previous positions
        self.fs = 1000.0  # Sample rate in Hz
        self.cutoff = 150.0  # Desired cutoff frequency in Hz
        self.order = 5  # Order of the filter
        self.b, self.a = self.design_filter(self.order, self.cutoff, self.fs)

        rospy.init_node("grub_pid")
        self.pub = rospy.Publisher("/robo_0/joint_0/pressure_command",
                                   PressureStamped,
                                   queue_size=1)

        self.pub1 = rospy.Publisher("/robo_0/joint_0/ctrl_info",
                                    JointControllerState,
                                    queue_size=1)

        self.alpha = 1  # adjust this value as needed
        self.filtered_value = 0  # initial value

        for i in range(10):
            init = PressureStamped()
            init.header.stamp = rospy.Time.now()
            init.pressure = [0, 0, 0, 0]
            self.pub.publish(init)
            rospy.sleep(0.5)

        self.sub = rospy.Subscriber("/robo_0/joint_0/angle_state",
                                    JointState,
                                    callback=self.callback)
        self.sub_cmd = rospy.Subscriber("/joint_angle_command", Float64,
                                        self.cmd_callback)

        self.ctrl_hist = []
        self.p_hist = []
        self.i_hist = []
        self.d_hist = []
        self.pavg = 200

    def update_setpoint(self):
        # Get the current time
        current_time = time.time()

        # Calculate the angle for the sine function
        angle = (current_time * 2 * np.pi) / 5

        # Calculate the new setpoint
        new_setpoint = np.sin(angle)

        # Apply prefilter to the new setpoint
        self.filtered_value = self.alpha * new_setpoint + (
            1 - self.alpha) * self.filtered_value

        # Update PID setpoint based on the filtered value
        self.pid.setpoint = self.filtered_value

    def cmd_callback(self, msg):
        # Apply prefilter to the joint angle command
        self.filtered_value = self.alpha * msg.data + (
            1 - self.alpha) * self.filtered_value

        # Update PID setpoint based on the filtered value
        self.pid.setpoint = self.filtered_value

    def design_filter(self, order, cutoff, fs):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        print(f"b: {b}, a: {a}")
        return b, a

    def plot(self):
        # plot the p, i, d components of the PID controller
        import matplotlib.pyplot as plt
        plt.plot(self.ctrl_hist, label="Control")
        plt.plot(self.p_hist, label="P")
        plt.plot(self.i_hist, label="I")
        plt.plot(self.d_hist, label="D")
        plt.legend()
        plt.show()

    def low_pass_filter(self, position):
        return position
        self.prev_positions.append(position)
        filtered_position = lfilter(self.b, self.a, list(self.prev_positions))
        return filtered_position[-1]

    def callback(self, msg):
        # filtered_position = self.low_pass_filter(msg.position[0])
        # self.update_setpoint()
        ctrl = self.pid(msg.position[0])
        ctrl2 = self.pid2(msg.position[1])
        # d_custom = self.kd * (msg.velocity[0])
        # ctrl -= d_custom

        p0, p1 = self.torque2pressures(ctrl)
        p2, p3 = self.torque2pressures(ctrl2)
        p, i, d = self.pid.components
        # print(
        #     f"p: {p:.3f} i: {i:.3f} d: {d_custom:.3f}, error: {(self.pid.setpoint - msg.position[0]):.3f}"
        # )

        # print(f"ctrl: {ctrl:.3f}, p0: {p0:.3f}, p1: {p1:.3f}")

        ctrl_msg = JointControllerState()
        ctrl_msg.header.stamp = rospy.Time.now()
        ctrl_msg.p = p
        ctrl_msg.i = i
        # ctrl_msg.d = d_custom
        ctrl_msg.d = d

        msg = PressureStamped()
        msg.header.stamp = rospy.Time.now()
        msg.pressure = [p0, p1, p2, p3]

        self.p_hist.append(p)
        self.i_hist.append(i)
        # self.d_hist.append(d_custom)
        self.d_hist.append(d)
        self.ctrl_hist.append(ctrl)
        self.pub.publish(msg)
        self.pub1.publish(ctrl_msg)

    def torque2pressures(self, torque):
        # convert torque to pressures
        p0 = self.pavg + torque
        p1 = self.pavg - torque

        return p0, p1


if __name__ == "__main__":
    controller = GrubPIDController()
    while not rospy.is_shutdown():
        rospy.spin()

    # controller.plot()
'''
Observations:

1. P gain by itself can get up to about .35 before osciallations become really bad. The proportional gain does
help decrease steady state error up to about here though. 
2. There's an interesting wiggle on the way up that is definitely at least 3rd order. 
3. The D gain makes everything worse. Oscillations get worse than just P control. This is because the 
velocity is highly noisy. So I should probably smooth this out before giving it to the PID controller.
update: The derivative control is bouncing around a lot. Smoothing certainly helps, but its still not great.
4. Just PI controller does the best. Little oscillations on the rising edge, small SSE, no overshoot. 
5. same thing happens when using the velocity out of the vive puck, lots of oscillations. 
6. we aren't saturating control either. 
6. could oscillations be because the error still bounces around before getting to the setpoint? idk... 
if ydot is bouncing around, then the derivative control will be bouncing around. So something causes P
to bounce, and derivative contorl exacerbates it. 
7. there's some high frequency bouncing that the vive picks up. It certainly hurts the propotional control
if the gain is too high. I can flick the grub and start the oscillations. 
8. heavier filtering helps alot on the the proprtional oscillations, but the filter is acting strangly. 
9. trying torque control doesn't really help. Very similar behavior with limit cycle. 
'''
