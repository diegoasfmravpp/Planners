import numpy as np
import shapely

def setup_ax(ax, title, configurationspace, start_state, target_state):
        """Sets up the axis for plotting."""
        ax.set_xlabel(r'$x(m)$')
        ax.set_ylabel(r'$y(m)$')
        ax.set_xlim(configurationspace.x_bounds)
        ax.set_ylim(configurationspace.y_bounds)
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')
        # plot goal configuration
        if target_state is not None:
            ax.arrow(target_state[0], target_state[1], 1.0*np.cos(target_state[2]), 
                1.0*np.sin(target_state[2]), color='red', width=.15, zorder=1e4)
        # plot start configuration
        if start_state is not None:
            poly = configurationspace.car_shape(start_state)
            shapely.plotting.plot_polygon(poly, ax, color='green', add_points=False, linewidth=1)
        # plot obstacles
        for obs in configurationspace.obstacles:
            shapely.plotting.plot_polygon(obs, ax, color='black', add_points=False, alpha=0.5)

def plot_shape_at_state(ax, configurationspace, state, color='blue'):
        poly = configurationspace.car_shape(state)
        if poly is not None:
            shapely.plotting.plot_polygon(poly,ax,color=color,add_points=False,linewidth=1)