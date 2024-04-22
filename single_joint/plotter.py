import matplotlib.pyplot as plt
from collections import deque


class JointAnglePlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'r-')
        self.x = deque(maxlen=100)
        self.y = deque(maxlen=100)
        self.ax.set_ylim(-1, 1)

    def update(self, data):
        self.x.append(data.time)
        self.y.append(data.sensor('vive_tracker').data[0])
        self.line.set_ydata(self.y)
        self.line.set_xdata(self.x)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show(self):
        plt.show(block=False)

    def close(self):
        plt.close(self.fig)
