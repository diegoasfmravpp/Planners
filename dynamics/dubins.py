import yaml
import casadi as ca
import os
import numpy as np
import shapely
from typing import List, Optional

class VehicleStateSpace:
    def __init__(self):

        # Load YAML file
        this_dir = os.path.dirname(__file__)
        yaml_path = os.path.join(this_dir, 'limits.yaml')
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)
        # Load vehicle limits
        v_max = data["v_max"]
        v_min = data["v_min"]
        max_wheel_angle = data["max_wheel_angle"]
        ax_max = data["ax_max"]
        ax_min = data["ax_min"]
        ay_max = data["ay_max"]
        max_steering_rate = data["max_steering_rate"]
        wheelbase = data["wheelbase"]

        self.wheelbase = wheelbase
        self.v_bounds = [v_min, v_max]
        self.ax_bounds = [ax_min, ax_max]
        self.ay_bounds = [-ay_max, ay_max]
        self.psi_bounds = [-max_wheel_angle, max_wheel_angle]
        self.phi_bounds = [-max_steering_rate, max_steering_rate]
        
    # Dynamics
    def x_dot(self, x, u):
        dx = x[3] * ca.cos(x[2])
        dy = x[3] * ca.sin(x[2])
        dtheta = (x[3]/self.wheelbase)*np.tan(x[4]) # steering angle (phi)
        dv = u[0]  # acceleration (a)
        dphi = u[1] # steering angle rate (psi)
        
        return ca.vertcat(dx, dy, dtheta, dv, dphi)
    
class VehicleConfigurationSpace:
    def __init__(self, dynamics : VehicleStateSpace,
                 x_bound : tuple,
                 y_bound : tuple,
                 car_reference_shape : shapely.geometry.Polygon,
                 obstacles : List[shapely.geometry.Polygon] = [],
                 distance_weights : Optional[list] = None):
        self.dynamics = dynamics
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.car_reference_shape = car_reference_shape
        self.obstacles = obstacles
        if distance_weights is None:
            self.distance_weights = [0]*4
        else:
            self.distance_weights = distance_weights

    def car_shape(self, state : list) -> shapely.geometry.Polygon:
        x, y, theta, v  = state
        #SOLUTION
        if self.car_reference_shape is None:
            return None
        x,y,theta,v,phi = state
        R = np.array([[np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]])
        car_polygon = shapely.transform(self.car_reference_shape, lambda pts: np.dot(R, pts.T).T + np.array([x, y]))
        return car_polygon