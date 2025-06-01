from .rrt import RRT
import matplotlib.pyplot as plt
import shapely
import numpy as np

def run_rrt(configurationspace, start_state, target_state, num_iters=1000, goal_sampling_bias=0.0):
    rrt = RRT(configurationspace)
    rrt.initialize(start_state, target_state)

    #plotting
    plt.ion()
    plt.show()
    def draw_tree():
        """Redraws the obstacles, goal, and tree on the current axis."""
        plt.gca().set_aspect('equal', 'box')
        plt.xlabel(r'$x(m)$')
        plt.ylabel(r'$y(m)$')
        plt.xlim(configurationspace.x_bounds)
        plt.ylim(configurationspace.y_bounds)
        # plot goal configuration
        if target_state is not None:
            plt.arrow(target_state[0], target_state[1], 1.0*np.cos(target_state[2]), 
                1.0*np.sin(target_state[2]), color='red', width=.15, zorder=1e4)
        # plot obstacles
        for obs in configurationspace.obstacles:
            shapely.plotting.plot_polygon(obs, plt.gca(), color='black', add_points=False, alpha=0.5)
        rrt.plot_mpl(plt.gca())

    #run RRT
    goal_node = None
    closest_node = None
    closest_dist = float('inf')
    collision_states = []
    for iteration in range(num_iters):    
        new_node, collision_state = rrt.extend(target_state, goal_sampling_bias)
     
        #check goal condition
        if new_node is not None:
            new_node_distance = configurationspace.distance(new_node.state,target_state)
            if new_node_distance < 0.5:
                print("Goal reached on iteration {}!".format(iteration))
                goal_node = new_node
                break
            if new_node_distance < closest_dist:
                closest_node = new_node
                closest_dist = new_node_distance
        collision_states.append(collision_state)
        #redraw plot periodically
        #if iter_idx % 50 == 0:
        if iteration % 50 == 0:
            plt.gca().cla()
            draw_tree()
            
            plt.draw()
            plt.pause(0.001)
    #final plot
    plt.ioff() # to keep plot on
    plt.gca().cla()
    draw_tree()
    if goal_node is not None:
        rrt.plot_path(plt.gca(), goal_node, color='red', lw=2)
    elif closest_node is not None:
        print("Closest distance to goal: {}".format(closest_dist))
        rrt.plot_path(plt.gca(), closest_node, color='orange', lw=2)
    for c in collision_states:
        if not(c is None):
            shapely.plotting.plot_polygon(configurationspace.car_shape(c), color='red')
    plt.draw()
    plt.show()