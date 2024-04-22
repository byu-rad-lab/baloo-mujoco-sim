"""
Curtis Johnson - 4/22/24

Simple maplotlib plotter to do live plotting from mujoco.

This is pretty slow still, but Im not sure its worth the time to speed up.

I looked into adding plotting directly into the mujoco gui, but this seemed somewhat difficult
without writing my own viewer. I just want to keep the default, so here we are.

I also looked at pyqtgraph since its faster, but the Qt has issues with all of the threads
already running. 

"""

import matplotlib.pyplot as plt
from collections import deque
import pyqtgraph as pg


class JointAnglePlotter:
    def __init__(self, max_len=500):
        self.fig, self.ax = plt.subplots(2, 1,
                                         sharex=True)  # Create 2x1 subplots
        self.u_line, = self.ax[0].plot([], [], 'r-', label='u (rad)')
        self.v_line, = self.ax[0].plot([], [], 'b-', label='v (rad)')
        self.udot_line, = self.ax[1].plot([], [], 'r-', label='udot (rad/s)')
        self.vdot_line, = self.ax[1].plot([], [], 'b-', label='vdot (rad/s)')

        self.x = deque(maxlen=max_len)
        self.y_u = deque(maxlen=max_len)
        self.y_v = deque(maxlen=max_len)
        self.y_udot = deque(maxlen=max_len)
        self.y_vdot = deque(maxlen=max_len)

        self.ax[0].set_ylim(-1.57, 1.57)
        self.ax[1].set_ylim(-5, 5)
        self.ax[1].set_xlabel('Sim Time (s)')
        for ax in self.ax:
            ax.grid(True)
            ax.legend()

    def update(self, data):
        self.x.append(data.time)
        self.y_u.append(data.sensor('vive_tracker').data[0])
        self.y_v.append(data.sensor('vive_tracker').data[1])
        self.y_udot.append(data.sensor('vive_tracker').data[2])
        self.y_vdot.append(data.sensor('vive_tracker').data[3])

        self.u_line.set_ydata(self.y_u)
        self.u_line.set_xdata(self.x)
        self.v_line.set_ydata(self.y_v)
        self.v_line.set_xdata(self.x)
        self.udot_line.set_ydata(self.y_udot)
        self.udot_line.set_xdata(self.x)
        self.vdot_line.set_ydata(self.y_vdot)
        self.vdot_line.set_xdata(self.x)

        for ax in self.ax:
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show(self):
        plt.show(block=False)

    def close(self):
        plt.close(self.fig)
