import yaml
import casadi as ca
import os


class Vehicle_Dynamics:
    def __init__(self):
        # Vehicle dynamics limits
        # v_max  = settings.get('vehicle.limits.max_speed')
        # ax_max = settings.get('vehicle.limits.max_longitudinal_acceleration')
        # ax_min = settings.get('vehicle.limits.min_longitudinal_acceleration')
        # ay_max = settings.get('vehicle.limits.max_lateral_acceleration')
        # max_steering_rate = settings.get('vehicle.limits.max_steering_rate')
        # wheelbase = settings.get('vehicle.geometry.wheelbase')
        # max_wheel_angle = settings.get('vehicle.geometry.max_wheel_angle')

        # max_steering_rate = steer2front(max_steering_rate)

        #hard coded, remove

        # Load YAML file
        this_dir = os.path.dirname(__file__)
        yaml_path = os.path.join(this_dir, 'limits.yaml')
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)

        v_max = data["v_max"]
        max_wheel_angle = data["max_wheel_angle"]
        ax_max = data["ax_max"]
        ax_min = data["ax_min"]
        ay_max = data["ay_max"]
        max_steering_rate = data["max_steering_rate"]
        wheelbase = data["wheelbase"]


        self.v_bounds = [0, v_max]
        self.theta_bounds = [-max_wheel_angle, max_wheel_angle]

        self.ax_bounds = [ax_min, ax_max]
        self.ay_bounds = [-ay_max, ay_max]

        self.omega_bounds = [-max_steering_rate, max_steering_rate]
        
    # Dynamics
    def x_dot(self, x, u):
        dx = x[3] * ca.sin(x[2])
        dy = x[3] * ca.cos(x[2])
        dtheta = u[1]
        dv = u[0]

        return ca.vertcat(dx, dy, dtheta, dv)