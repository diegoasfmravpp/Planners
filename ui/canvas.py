from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import random
import numpy as np
import shapely.geometry
from utils.plotting import setup_ax, plot_shape_at_state


class ConfigSpaceCanvas(FigureCanvas):
    def __init__(self, configurationspace):
        self.fig = Figure()
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

        self.configurationspace = configurationspace

        self.x0 = None
        self.xf = None

        self.rand_start()
        self.rand_goal()

        self.ax.clear()
        setup_ax(self.ax, "Plot", self.configurationspace, self.x0, self.xf)
        self.draw()

    def rand_start(self):
        state = self.configurationspace.sample_state()
        while np.any(self.configurationspace.car_shape(state).intersects(self.configurationspace.obstacles)):
            state = self.configurationspace.sample_state()
        self.x0 = state

        self.ax.clear()
        setup_ax(self.ax, "Plot", self.configurationspace, self.x0, self.xf)
        self.draw()

    def rand_goal(self):
        state = self.configurationspace.sample_state()
        while np.any(self.configurationspace.car_shape(state).intersects(self.configurationspace.obstacles)):
            state = self.configurationspace.sample_state()
        self.xf = state

        self.ax.clear()
        setup_ax(self.ax, "Plot", self.configurationspace, self.x0, self.xf)
        self.draw()

    def rand_obstacles(self, n=5):
        self.configurationspace.obstacles = []
        obstacles = []
        #obstacle setup
        for i in range(n):
            while True:
                c1 = np.random.uniform(-10,10,2)
                center = shapely.geometry.Point(c1[0],c1[1])
                radius = np.random.uniform(1,2)
                obstacle = center.buffer(radius)
                if not np.any(obstacle.intersects([self.configurationspace.car_shape(self.x0),
                                               self.configurationspace.car_shape(self.xf)])):
                    break
            obstacles.append(obstacle)
        self.configurationspace.obstacles = obstacles

        self.ax.clear()
        setup_ax(self.ax, "Plot", self.configurationspace, self.x0, self.xf)
        self.draw()

