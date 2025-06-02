import yaml
import casadi as ca
import numpy as np
import shapely
from typing import List, Optional
from utils import settings

class VehicleStateSpace:
    def __init__(self, t_step: float = 0.1):


        data = settings.get('dynamics.limits')
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

        self.t_step = t_step
        
    # Dynamics
    def x_dot(self, x, u, casadi=False):
        if casadi:
            # Use CasADi for symbolic computation
            dx = x[3] * ca.cos(x[2])
            dy = x[3] * ca.sin(x[2])
            dtheta = (x[3]/self.wheelbase)*ca.tan(x[4])
            dv = u[0]  # acceleration (a)
            dphi = u[1] # steering angle rate (psi)
            return ca.vertcat(dx, dy, dtheta, dv, dphi)
        else:
        # Use NumPy for numerical computation
            dx = x[3] * np.cos(x[2])
            dy = x[3] * np.sin(x[2])
            dtheta = (x[3]/self.wheelbase)*np.tan(x[4])  # steering angle (phi)
            dv = u[0]  # acceleration (a)
            dphi = u[1] # steering angle rate (psi)
            return np.array([dx, dy, dtheta, dv, dphi])
        
    def state_in_bounds(self, state):
        """Check if velocity and steering angle are within bounds."""
        x, y, theta, v, phi = state
        return (self.v_bounds[0] <= v <= self.v_bounds[1] and
                self.phi_bounds[0] <= phi <= self.phi_bounds[1])
    def control_in_bounds(self, control):
        """Check if control inputs are within bounds."""
        a, psi = control
        return (self.ax_bounds[0] <= a <= self.ax_bounds[1] and
                self.psi_bounds[0] <= psi <= self.psi_bounds[1])
    
    def sample_state(self, x_bounds, y_bounds, xf=None, goal_sampling_bias=None):
        """Sample a state optionally biased towards xf"""
        if xf is not None and goal_sampling_bias is not None:
            if np.random.rand() < goal_sampling_bias:
                return (xf)
        x = np.random.uniform(*x_bounds)
        y = np.random.uniform(*y_bounds)
        theta = np.random.uniform(0, 2*np.pi)
        v = np.random.uniform(*self.v_bounds)
        phi = np.random.uniform(*self.phi_bounds)
        return np.array([x, y, theta, v, phi])
        
    def sample_control(self):
        """Sample a control input"""
        a = np.random.uniform(*self.ax_bounds)
        psi = np.random.uniform(*self.psi_bounds)
        return np.array([a, psi])
    
    def step(self, state, control, dt=None):
        """
        Step the vehicle dynamics using Euler integration.
        Input:
            state: list of 5 floats, [x, y, theta, v, phi]
            control: list of 2 floats, [a, psi]
            dt: float, time step for integration
        Output:
            new_state: list of 5 floats, updated state
        """
        if dt is None:
            dt = self.t_step
        dx, dy, dtheta, dv, dphi = self.x_dot(state, control)
        x, y, theta, v, phi = state
        x += dx * dt
        y += dy * dt
        theta += dtheta * dt
        v += dv * dt
        phi += dphi * dt
        # Normalize theta to be within [-pi, pi]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        return [x, y, theta, v, phi]
    
    def integrate(self, state, control, T, dt=None):
        """
        Integrate the vehicle dynamics over a time period T with a given control input.
        Input:
            state: list of 5 floats, [x, y, theta, v, phi]
            control: list of 2 floats, [a, psi]
            T: float, total time to integrate over
            dt: float, time step for integration (optional, defaults to self.t_step)
        Output:
            states: list of states at each time step
        """
        if dt is None:
            dt = self.t_step
        states = [state]
        t = 0.0
        while t+dt <= T:
            state = self.step(state, control, dt)
            if not (self.state_in_bounds(state) and self.control_in_bounds(control)):
                break
            states.append(state)
            t += dt
        # Handle the last step if T is not an exact multiple of dt
        if t < T:
            state = self.step(state, control, T - t)
            if self.state_in_bounds(state) and self.control_in_bounds(control):
                states.append(state)
        return states
    
class VehicleConfigurationSpace:
    def __init__(self, dynamics : VehicleStateSpace,
                 x_bounds : tuple,
                 y_bounds : tuple,
                 car_reference_shape : shapely.geometry.Polygon,
                 obstacles : List[shapely.geometry.Polygon] = [],
                 distance_weights : Optional[list] = None):
        self.dynamics = dynamics
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.car_reference_shape = car_reference_shape
        self.obstacles = obstacles
        if distance_weights is None:
            self.distance_weights = [0]*4
        else:
            self.distance_weights = distance_weights
    
    def sample_state(self, xf=None, goal_sampling_bias=None) -> list:
        """Sample a state optionally biased towards xf"""
        return self.dynamics.sample_state(self.x_bounds, self.y_bounds, xf, goal_sampling_bias)

    def car_shape(self, state : list) -> shapely.geometry.Polygon:
        #SOLUTION
        if self.car_reference_shape is None:
            return None
        x,y,theta,v,phi = state
        R = np.array([[np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]])
        car_polygon = shapely.transform(self.car_reference_shape, lambda pts: np.dot(R, pts.T).T + np.array([x, y]))
        return car_polygon
    
    def distance(self, state1 : list, state2 : list) -> float:
        diff = np.array(state1) - np.array(state2)
        theta_diff = state2[2]-state1[2]
        if theta_diff > np.pi:
            theta_diff = theta_diff - 2*np.pi
        elif theta_diff <= -np.pi:
            theta_diff = theta_diff + 2*np.pi
        theta_diff = abs(theta_diff)
        diff[2] = theta_diff
        weighted_diff = self.distance_weights*diff**2
        return np.sqrt(np.sum(weighted_diff))
    
    def state_in_bounds(self, state : list) -> bool:
        """Check if the full state is within the bounds of the configuration space."""
        return (self.dynamics.state_in_bounds(state) and 
                self.x_bounds[0] <= state[0] <= self.x_bounds[1] and 
                self.y_bounds[0] <= state[1] <= self.y_bounds[1])
    
    def state_valid(self, state : list) -> bool:
        """State feasibility checking."""
        if not self.state_in_bounds(state):
            return False
        # Check if the car shape intersects with any obstacles
        if np.any(self.car_shape(state).intersects(self.obstacles)):
            return False
        # Check if the car is able to brake without hitting an obstacle
        brake_d = state[3]**2/(2*self.dynamics.ax_bounds[0])
        if brake_d > 0:
            brake_x = state[0] - brake_d*np.cos(state[2])
            brake_y = state[1] - brake_d*np.sin(state[2])
            new_state = [brake_x,brake_y,state[2],state[3],state[4]]
            if np.any(self.car_shape(new_state).intersects(self.obstacles)):
                return False
        return True