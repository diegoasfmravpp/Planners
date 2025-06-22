from .rrt import RRT
import matplotlib.pyplot as plt
import shapely
import numpy as np

def run_rrt(configurationspace, start_state, target_state, settings):
    rrt = RRT(configurationspace, settings)
    rrt.initialize(start_state, target_state)

    def setup_ax(ax, title):
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
        # plot obstacles
        for obs in configurationspace.obstacles:
            shapely.plotting.plot_polygon(obs, ax, color='black', add_points=False, alpha=0.5)
    
    def draw_collision_states(ax):
        """Draws the collision states on the current axis."""
        num_collisions = 0
        for c in collision_states:
            if not(c is None):
                shapely.plotting.plot_polygon(configurationspace.car_shape(c), ax, color='red', add_points=False, linewidth=1)
                num_collisions += 1
        return num_collisions
    
    if settings['plotting']:
        #plotting
        plt.ion()
        plt.show()
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2 rows, 2 columns
        fig.subplots_adjust(hspace=0.3)
        # setup axes
        titles = ["RRT Demo", "RRT Tree", "Collision states avoided", "Goal Reached"]
        for i,ax in enumerate(axs.flat):
            setup_ax(ax, titles[i])

    #run RRT
    goal_node = None
    closest_node = None
    closest_dist = float('inf')
    collision_states = []
    for iteration in range(settings['max_iterations']):    
        new_node, collision_state = rrt.extend(target_state)
     
        #check goal condition
        if new_node is not None:
            new_node_distance = configurationspace.distance(new_node.state,target_state)
            if new_node_distance < 0.5:
                goal_node = new_node
                break
            if new_node_distance < closest_dist:
                closest_node = new_node
                closest_dist = new_node_distance
        collision_states.append(collision_state)
        #redraw plot periodically
        if settings['plotting'] and iteration % 100 == 0:
            axs[0, 0].cla()
            setup_ax(axs[0, 0],"RRT Demo")
            rrt.plot_trajectories(axs[0, 0])
            fig.canvas.draw()
            plt.pause(0.001)

    if settings['plotting']:
        #final plot
        plt.ioff() # to keep plot on
        axs[0, 0].cla()
        setup_ax(axs[0, 0],"RRT Demo")
        rrt.plot_trajectories(axs[0, 0])
        axs[0, 0].set_title("RRT Demo")
        rrt.plot_trajectories(axs[0, 1])
        axs[0, 1].set_title("RRT Tree")

        draw_collision_states(axs[0, 0])
        num_collisions = draw_collision_states(axs[1, 0])
        axs[1, 0].set_title("{} Collision states avoided".format(num_collisions))

        if goal_node is not None:
            rrt.plot_path(axs[0, 0], goal_node, color='red', lw=2)
            rrt.plot_path(axs[1, 1], goal_node, color='red', lw=2)
            axs[1, 1].set_title("Goal reached on iteration {}!".format(iteration))
        elif closest_node is not None:
            rrt.plot_path(axs[0, 0], closest_node, color='orange', lw=2)
            rrt.plot_path(axs[1, 1], closest_node, color='orange', lw=2)
            axs[1, 1].set_title("Closest distance to goal: {}".format(round(closest_dist, 3)))

        plt.draw()
        plt.show()

    return rrt, closest_node, goal_node
    