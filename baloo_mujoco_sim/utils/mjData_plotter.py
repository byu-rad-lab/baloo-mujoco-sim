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
from pyqtgraph import mkPen
from PyQt5 import QtWidgets
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject


class MjDataPlotter(QObject):
    data_updated = pyqtSignal(dict)

    def __init__(self, time_hist, sim_dt):
        super().__init__()
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget()

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
            for j in range(2):  # For each angle
                curves.append(
                    p.plot(
                        pen=mkPen(color=colors[j], style=QtCore.Qt.SolidLine),
                        name=f'angle{j}'))  # Add a unique name for the curve
                curves.append(
                    p.plot(pen=mkPen(color=colors[j],
                                     style=QtCore.Qt.DashLine),
                           name=f'angle{j}_cmd')
                )  # Add a unique name for the pressure command

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

        self.angle_cmd_data = []
        for i in range(3):
            self.angle_cmd_data.append(
                [deque(maxlen=max_len) for _ in range(2)])

        # Connect the signal to the slot
        self.data_updated.connect(self.save_data)

        # self.timer = QtCore.QTimer()
        # self.timer.timeout.connect(self.plot_data)
        # self.timer.start(1000)

    @pyqtSlot(dict)
    def save_data(self, data):
        # Code to save the data goes here...
        # print("Data saved!")
        self.plot_data()

    def plot_data(self):
        # print("Plotting data...")
        for joint_num in range(3):
            for angle_num in range(2):
                # Update the plot
                self.curves[joint_num][angle_num * 2].setData(
                    self.time_data, self.angle_data[joint_num][angle_num])

                self.curves[joint_num][angle_num * 2 + 1].setData(
                    self.time_data, self.angle_cmd_data[joint_num][angle_num])
        # self.plot_widget.replot()
        QtWidgets.QApplication.processEvents()  # you MUST process the plot now

    def update(self, mjmodel, mjdata, custom_data: dict = None):
        # Update data
        self.time_data.append(mjdata.time)

        for joint_num in range(3):
            angles = get_joint_angles(mjmodel, mjdata, 'left', joint_num)
            vel = get_joint_vel(mjmodel, mjdata, 'left', joint_num)
            for angle_num in range(2):
                self.angle_data[joint_num][angle_num].append(angles[angle_num])
                self.angle_cmd_data[joint_num][angle_num].append(
                    custom_data[f"joint{joint_num}_angle{angle_num}_cmd"])

                # # Update the plot
                # self.curves[joint_num][angle_num * 2].setData(
                #     self.time_data, self.angle_data[joint_num][angle_num])

                # self.curves[joint_num][angle_num * 2 + 1].setData(
                #     self.time_data, self.angle_cmd_data[joint_num][angle_num])

        # Emit the signal with the updated data
        self.data_updated.emit(custom_data)

        # QtWidgets.QApplication.processEvents()  # you MUST process the plot now

    def show(self):
        self.win.show()

    def close(self):
        self.win.close()
