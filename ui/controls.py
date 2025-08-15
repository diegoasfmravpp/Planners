from PyQt5.QtWidgets import QVBoxLayout, QLabel, QPushButton, QComboBox, QSpinBox

class ControlsPanel(QVBoxLayout):
    def __init__(self, run_callback, canvas):
        super().__init__()
        self.canvas = canvas

        self.addWidget(QLabel("Algorithm:"))
        self.algo_dropdown = QComboBox()
        self.algo_dropdown.addItems(["RRT", "PRM", "A*"])
        self.addWidget(self.algo_dropdown)

        self.addWidget(QLabel("Iterations:"))
        self.iter_spin = QSpinBox()
        self.iter_spin.setValue(500)
        self.addWidget(self.iter_spin)

        run_btn = QPushButton("Run Planner")
        run_btn.clicked.connect(lambda: run_callback(
            self.algo_dropdown.currentText(),
            {"iterations": self.iter_spin.value()}
        ))
        self.addWidget(run_btn)

        start_btn = QPushButton("Random Start")
        start_btn.clicked.connect(self.canvas.rand_start)
        self.addWidget(start_btn)

        goal_btn = QPushButton("Random Goal")
        goal_btn.clicked.connect(self.canvas.rand_goal)
        self.addWidget(goal_btn)

        obs_btn = QPushButton("Random Obstacles")
        obs_btn.clicked.connect(lambda: self.canvas.rand_obstacles(10))
        self.addWidget(obs_btn)

        self.addStretch()
