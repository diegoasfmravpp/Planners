from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import random
import numpy as np

class ConfigSpaceCanvas(FigureCanvas):
    def __init__(self, configurationspace):
        self.fig = Figure()
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

        self.configurationspace = configurationspace

        self.x0 = None
        self.xf = None
        self.obstacles = []
        self.path = []

        self.draw_space()

    def draw_space(self):
        self.ax.clear()
        self.ax.set_xlim(*self.configurationspace.x_bounds)
        self.ax.set_ylim(*self.configurationspace.y_bounds)
        self.ax.set_title("Configuration Space")

        if self.x0 is not None:
            self.ax.plot(self.x0[0], self.x0[1], 'go', label="Start")

        if self.xf is not None:
            self.ax.plot(self.xf[0], self.xf[1], 'ro', label="Goal")

        for obs in self.obstacles:
            circle = plt.Circle(obs, 0.3, color='black')
            self.ax.add_patch(circle)

        if self.path:
            xs, ys = zip(*self.path)
            self.ax.plot(xs, ys, 'b-', label="Path")

        self.ax.legend(loc="upper right")
        self.draw()

    def rand_start(self):
        state = self.configurationspace.sample_state()
        while np.any(self.configurationspace.car_shape(state).intersects(self.configurationspace.obstacles)):
            state = self.configurationspace.sample_state()
        self.x0 = state
        self.draw_space()

    def rand_goal(self):
        state = self.configurationspace.sample_state()
        while np.any(self.configurationspace.car_shape(state).intersects(self.configurationspace.obstacles)):
            state = self.configurationspace.sample_state()
        self.xf = state
        self.draw_space()

    def rand_obstacles(self, n=5):
        self.obstacles = [(random.uniform(*self.configurationspace.x_bounds),
                           random.uniform(*self.configurationspace.y_bounds)) for _ in range(n)]
        self.draw_space()

    def set_path(self, path):
        self.path = path
        self.draw_space()
