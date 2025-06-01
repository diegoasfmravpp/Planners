from __future__ import annotations
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import shapely.plotting
import numpy as np
import weakref
from typing import Tuple,List,Optional
from dynamics.dubins import VehicleStateSpace, VehicleConfigurationSpace
from .nearestneighbors import NearestNeighbors

class RRTNode:
    def __init__(self, state,
                parent : RRTNode = None,
                control_from_parent :list = None, dt_from_parent : float = None, traj_from_parent :List[list] = None):
        self.state = state
        if parent is None:
            self.parent = None
        else:
            self.parent = weakref.ref(parent)
        self.control_from_parent = control_from_parent
        self.dt_from_parent = dt_from_parent
        self.traj_from_parent = traj_from_parent
        self.children = []
    
    def path_from_root(self) -> List[RRTNode]:
        """Returns the path from the root to this node."""
        if self.parent is None:
            return [self]
        else:
            res = self.parent().path_from_root()
            res.append(self)
            return res
    
    def traj_from_root(self) -> Tuple[List[float],List[list],List[list]]:
        """Returns the trajectory from the root to this node.
        The path consists of (ts,xs,us) where ts are the times at which the
        states xs are reached, and us are the controls taken to reach the
        next state."""
        if self.parent is None:
            return [0], [self.state], []
        else:
            ts,xs,us = self.parent().traj_from_root()
            n = len(self.traj_from_parent)
            ts += np.linspace(ts[-1], ts[-1] + self.dt_from_parent, n).tolist()[1:]
            xs += self.traj_from_parent[1:]
            us += [self.control_from_parent]*(n-1)
            return ts,xs,us


class RRT:
    def __init__(self, configurationspace : VehicleConfigurationSpace):
        # self.delta = delta
        self.configurationspace = configurationspace
        self.nn = NearestNeighbors(configurationspace.distance, method='kdtree')
        self.nodes = []         # type : List[RRTNode]
        self.start_node = None        # type : RRTNode
        self.goal = None

    def initialize(self, start_state : list, target_state : list = None):
        """Initialize the RRT with a start and goal state."""
        self.start_node = RRTNode(start_state)
        self.nodes.append(self.start_node)
        self.nn.add(start_state,self.start_node)
        self.goal = target_state
    
    def extend(self, target_state, goal_sampling_bias) -> Optional[RRTNode]:
        """Extends the tree once"""
        sampled_state = self.configurationspace.sample_state(target_state, goal_sampling_bias)
        nearest_node = self.nearest(sampled_state)
        if nearest_node is None:
            return None
        nearest_state = nearest_node.state
        new_control, integrated_time, trajectory, collision_state = self.propagate_random_controls(nearest_state,target_state)
        if new_control is None:  #infeasible
            return None, collision_state
        new_state = trajectory[-1]
        new_node = RRTNode(new_state, nearest_node, new_control, integrated_time, trajectory)
        nearest_node.children.append(new_node)
        self.nodes.append(new_node)
        self.nn.add(new_state,new_node)   #add to nearest neighbors data structure
        return new_node, None

    def nearest(self, state) -> Optional[RRTNode]:
        """Brute force nearest neighbor search."""
        return self.nn.nearest(state)[1]
    
    def propagate_random_controls(self, state, target_state) -> Tuple[list,float,List[list]]:
        """Propagate the state with k random controls and return the control,
        time it was applied, and trajectory of the closest end state to the target state.
        """
        k = 1
        trajectories_end_distance = np.zeros(k)
        u = np.zeros([k,2])
        integration_time_limit = 1.0
        integration_times  = np.zeros(k)
        trajectories = []
        for i in range(k):
            u[i] = self.configurationspace.dynamics.sample_control()
            integration_times[i] = (np.random.random() + 1e-10) * integration_time_limit
            trajectories.append(self.configurationspace.dynamics.integrate(state, u[i], integration_times[i]))
            trajectories_end_distance[i] = self.configurationspace.distance(trajectories[i][-1],target_state)
        min_index = np.argmin(trajectories_end_distance)

        for state in trajectories[min_index][::-1]:
            if not self.configurationspace.state_valid(state):
                if np.any(self.configurationspace.car_shape(state).intersects(self.configurationspace.obstacles)):
                    return None,None,None,state
                else:
                    return None, None, None, None
        return u[min_index],integration_times[min_index],trajectories[min_index], None
    
    def plot_mpl(self, ax, colors = None):
        if colors is None: 
            colors = pl.cm.viridis(np.linspace(0, 1, len(self.nodes)))
        for idx,node in enumerate(self.nodes):
            x,y,theta,v,phi = node.state
            s = 0.08*v 
            ax.arrow(x, y, s*np.cos(theta), s*np.sin(theta), 
                        fc=colors[idx], lw=0.01, width=0.05)
            if node.parent is not None:
                traj = node.traj_from_parent
                x = [s[0] for s in traj]
                y = [s[1] for s in traj]
                ax.plot(x, y, color=colors[idx], linewidth=1)
            # if len(node.children) == 0:  #leaf node
            #    self.plot_state_mpl(ax, node.state, 'blue')
        self.plot_state_mpl(ax, self.start_node.state, 'green')

    def plot_path(self, ax, node : RRTNode, color='blue', lw=3):
        ts,xs,us = node.traj_from_root()
        x = [s[0] for s in xs]
        y = [s[1] for s in xs]
        ax.plot(x, y, color=color, lw=lw)
        nodes = node.path_from_root()
        for n in nodes:
            self.plot_state_mpl(ax, n.state, color=color)
    
    def plot_state_mpl(self, ax, state, color='blue'):
        poly = self.configurationspace.car_shape(state)
        if poly is not None:
            shapely.plotting.plot_polygon(poly,ax,color=color,add_points=False,linewidth=1)

    
    
    