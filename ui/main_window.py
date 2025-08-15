import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget
from .canvas import ConfigSpaceCanvas
from .controls import ControlsPanel
from sampling import run
from  utils import settings


# from planners import rrt, prm, astar  # assume you have these

class MainWindow(QMainWindow):
    def __init__(self, configurationspace):
        super().__init__()
        self.setWindowTitle("Motion Planner GUI")

        layout = QHBoxLayout()

        self.canvas = ConfigSpaceCanvas(configurationspace)
        self.controls = ControlsPanel(self.run_planner, self.canvas)

        layout.addWidget(self.canvas)
        layout.addLayout(self.controls)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def run_planner(self, algo_name, window_params):
        start = self.canvas.x0
        goal = self.canvas.xf
        params = []

        if algo_name == "RRT":
            params = settings.get('demo', 'rrt')
            for key, value in params.items():
                params[key] = window_params[key] if key in window_params else value
            print("Running RRT with params:", params)
            run.run_rrt(self.canvas, start, goal, params)
            # path = rrt.run_rrt(start, goal, obstacles, params)
        elif algo_name == "PRM":
            print("Running PRM with params:", window_params)
            # path = prm.run_prm(start, goal, obstacles, params)
        elif algo_name == "A*":
            print("Running A* with params:", window_params)
            # path = astar.run_astar(start, goal, obstacles, params)
        else:
            path = []

        # self.canvas.set_path(path)

def main():
    from config.dynamics.dubins import VehicleConfigurationSpace
    config_space = VehicleConfigurationSpace()  # create your config space

    app = QApplication(sys.argv)
    window = MainWindow(config_space)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
