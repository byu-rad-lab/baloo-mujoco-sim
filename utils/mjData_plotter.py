"""
# Curtis Johnson - 4/22/24

Simple maplotlib plotter to do live plotting from mujoco.

"""
import time

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
from collections import deque
from utils.baloo_mj_api import get_joint_pressures, get_joint_vel, get_joint_angles


class MjDataPlotter():
    def __init__(self, time_hist, sim_dt):
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title="Joint angles")

        self.plots = []
        self.curves = []
        colors = ['y', 'r', 'g', 'b']

        for i in range(3):  # For each joint
            p = self.win.addPlot()
            p.addLegend()  # Add a legend to the plot
            p.setLabel('left', f'Angle for Joint {i}', units='rad')
            p.setYRange(-1.5, 1.5)
            p.showGrid(x=True, y=True)

            curves = []
            for j in range(2):  # For each pressure p0-p3
                curves.append(p.plot(
                    pen=colors[j],
                    name=f'angle{j}'))  # Add a unique name for the curve
                # curves.append(p.plot(pen='r', name=f'p{j}_cmd')
                #   )  # Add a unique name for the pressure command

            self.plots.append(p)
            self.curves.append(curves)

            if i < 2:  # Don't add a new row after the last plot
                self.win.nextRow()

        max_len = int(time_hist / sim_dt)

        self.time_data = deque(maxlen=max_len)
        # self.angle_data[joint_num][angle_num]
        self.angle_data = []
        for i in range(3):
            self.angle_data.append([deque(maxlen=max_len) for _ in range(2)])

    def update(self, mjmodel, mjdata, custom_data: dict = None):
        # Update data
        self.time_data.append(mjdata.time)

        # start = time.time()
        for joint_num in range(3):
            angles = get_joint_angles(mjmodel, mjdata, 'left', joint_num)
            vel = get_joint_vel(mjmodel, mjdata, 'left', joint_num)
            for angle_num in range(2):
                self.angle_data[joint_num][angle_num].append(angles[angle_num])
                self.curves[joint_num][angle_num].setData(
                    self.time_data, self.angle_data[joint_num][angle_num])
        # print(time.time() - start)

        QtGui.QApplication.processEvents()  # you MUST process the plot now

    def show(self):
        self.win.show()

    def close(self):
        self.win.close()
