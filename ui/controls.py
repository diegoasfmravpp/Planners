from PyQt5.QtWidgets import QVBoxLayout, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox

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
        self.iter_spin.setRange(1, 10000)
        self.iter_spin.setValue(1000)
        self.iter_spin.setSingleStep(100)
        self.addWidget(self.iter_spin)

        self.addWidget(QLabel("Goal Bias:"))
        self.bias_spin = QDoubleSpinBox()
        self.bias_spin.setRange(0, 1)
        self.bias_spin.setValue(0.5)
        self.bias_spin.setSingleStep(0.1)
        self.addWidget(self.bias_spin)

        self.addWidget(QLabel("Controls Sampled:"))
        self.controls_spin = QSpinBox()
        self.controls_spin.setRange(1, 10)
        self.controls_spin.setValue(3)
        self.controls_spin.setSingleStep(1)
        self.addWidget(self.controls_spin)

        run_btn = QPushButton("Run Planner")
        run_btn.clicked.connect(lambda: run_callback(
            self.algo_dropdown.currentText(),
            {"max_iterations": self.iter_spin.value(),
             "goal_sampling_bias": self.bias_spin.value(),
             "sampled_controls": self.controls_spin.value()}
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
