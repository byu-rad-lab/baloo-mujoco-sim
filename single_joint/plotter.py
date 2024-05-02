"""
# Curtis Johnson - 4/22/24

Simple maplotlib plotter to do live plotting from mujoco.

"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
from collections import deque


class JointAnglePlotter():
    def __init__(self, time_hist, sim_dt):
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title="Joint angles")

        # Create first plot and curves
        p1 = self.win.addPlot()
        p1.addLegend()  # Add a legend to the plot
        p1.setLabel('left', 'Joint Angle', units='rad')
        p1.setYRange(-np.pi, np.pi)
        p1.showGrid(x=True, y=True)
        curve1 = [
            p1.plot(pen='y', name='u'),  # Add a unique name for the curve
            p1.plot(pen='r', name='v')  # Add a unique name for the curve
        ]
        self.win.nextRow()

        # Create second plot and curves
        p2 = self.win.addPlot()
        p2.setXLink(p1)  # Link the x-axis of the second plot to the first plot
        p2.addLegend()  # Add a legend to the plot
        p2.setLabel('left', 'Joint Velocity', units='rad/s')
        p2.setLabel('bottom', 'Sim Time', units='s')
        p2.setYRange(-5, 5)
        p2.showGrid(x=True, y=True)
        curve2 = [
            p2.plot(pen='y', name='udot'),  # Add a unique name for the curve
            p2.plot(pen='r', name='vdot')  # Add a unique name for the curve
        ]

        self.plots = [p1, p2]
        self.curves = [curve1, curve2]

        max_len = int(time_hist / sim_dt)
        self.udata = deque(maxlen=max_len)
        self.vdata = deque(maxlen=max_len)
        self.udotdata = deque(maxlen=max_len)
        self.vdotdata = deque(maxlen=max_len)
        self.timedata = deque(maxlen=max_len)

    def update(self, mjmodel, mjdata):
        # Update data
        self.udata.append(mjdata.sensor("left_0").data[0])
        self.vdata.append(mjdata.sensor("left_0").data[1])
        self.udotdata.append(mjdata.sensor("left_0").data[2])
        self.vdotdata.append(mjdata.sensor("left_0").data[3])
        self.timedata.append(mjdata.time)

        # Update plots
        self.curves[0][0].setData(self.timedata, self.udata)
        self.curves[0][1].setData(self.timedata, self.vdata)

        self.curves[1][0].setData(self.timedata, self.udotdata)
        self.curves[1][1].setData(self.timedata, self.vdotdata)

        QtGui.QApplication.processEvents()  # you MUST process the plot now

    def show(self):
        self.win.show()

    def close(self):
        self.win.close()
