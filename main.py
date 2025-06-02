import sys
import numpy as np
import shapely
import shapely.geometry
from shapely.geometry import Polygon
from shapely import plotting
import matplotlib.pyplot as plt
from config.dynamics.dubins import VehicleStateSpace, VehicleConfigurationSpace
from optimization.casadi_planning import cone_trajectory_optimization, trajectory_optimization
from sampling.run import run_rrt
from  utils import settings

def setup_system():
    t_step, car_L = 0.1, 0.5

    t_bound = [0.0, 1.0]
    x_bound = [-10.0, 10.0]
    y_bound = [-10.0, 10.0]

    dynamics = VehicleStateSpace()

    car_w, car_h = 0.75, 0.3

    dist_weights = np.array([1.0, 1.0, 0.1, 2.0, 2.0])
    car_shape = None

    car_points = np.array([[-car_w*0.1, -car_h/2], [ car_w*0.9, -car_h/2], 
                            [ car_w*0.9,  car_h/2], [-car_w*0.1,  car_h/2]])
    car_shape = shapely.geometry.Polygon(car_points)
    obstacles = []

    #obstacle setup
    for i in range(10):
        c1 = np.random.uniform([-5,10])
        np.random.shuffle(c1)
        center = shapely.geometry.Point(c1[0],c1[1])
        radius = np.random.uniform([1,2])
        np.random.shuffle(radius)
        obstacle = center.buffer(radius[0])
        obstacles.append(obstacle)
    # obs1 = shapely.geometry.Point(-1.2,4).buffer(1)
    # obs2 = shapely.geometry.Point(1.2,9).buffer(2)
    # obstacles = [obs1,obs2]
    dubins = VehicleConfigurationSpace(dynamics, x_bound, y_bound, car_shape, obstacles, dist_weights)
    return dubins

def plot_trajs(ts,xs,us,ax, **kwargs):
    assert len(ts) == len(xs)
    assert len(ts) == len(us)+1
    ax.plot(ts,[x[0] for x in xs],label='x',**kwargs)
    ax.plot(ts,[x[1] for x in xs],label='y',**kwargs)
    ax.plot(ts,[np.mod(x[2]+np.pi,np.pi*2)-np.pi for x in xs],label='theta',**kwargs)
    ax.plot(ts,[x[3] for x in xs],label='v',**kwargs)
    ax.plot(ts,[x[4] for x in xs],label='phi',**kwargs)
    ax.plot(ts[:-1],[u[0] for u in us],label='a',**kwargs)
    ax.plot(ts[:-1],[u[1] for u in us],label='psi',**kwargs)
    ax.legend()

def optimization_demo():
    dubins = setup_system()

    x0 = np.array([0]*5)
    # x0[2] = np.pi/2  # initial orientation
    # xtarget = np.array([0,10,np.pi/2,0,0])
    xf = np.array([0, 3,np.pi,0,0])
    N = 100
    dt = 0.1

    #first argument can be 'x0', 'line', or a list of T+1 states
    #second argument can be 'zero', 'random', or a list of T controls
    xt,ut,ft = trajectory_optimization(dubins,x0,xf,N,dt,x_guess=None, u_guess=None)
    
    print("Objective function",ft)

    fig,axs = plt.subplots(1,2,figsize=(10,4))
    # workspace trajectory
    # axs[0].plot([x[0] for x in xt],[x[1] for x in xt])
    axs[0].plot([x[0] for x in xt],[x[1] for x in xt],linestyle=':')

    # plot goal configuration
    axs[0].arrow(xf[0], xf[1], 1.0*np.cos(xf[2]), 
        1.0*np.sin(xf[2]), color='red', width=.15, zorder=1e4)
    # plot obstacles
    for obs in dubins.obstacles:
        shapely.plotting.plot_polygon(obs, axs[0], color='black', add_points=False, alpha=0.5)

    axs[0].axis('equal')
    plot_trajs(list(range(N+1)),xt,ut,axs[1])
    plt.show()

def test_1():
    cones = [
                {'x': 0.0, 'y': 15.0, 'orientation': 'left'},
                {'x': 0.0, 'y': 30.0, 'orientation': 'right'},
                {'x': 0.0, 'y': 45.0, 'orientation': 'left'},
                {'x': 0.0, 'y': 60.0, 'orientation': 'standing'},
                {'x': 0.0, 'y': 75.0, 'orientation': 'right'},
                {'x': 0.0, 'y': 90.0, 'orientation': 'left'},
            ]
    
    vehicle = VehicleStateSpace()
    x0 = np.array([0.0,0.0,0.0,0.0])
    xf = np.array([0.0,100.0,0.0,0.0])
    N = 200
    dt = 0.1
    X, U, _, x_guess = cone_trajectory_optimization(vehicle=vehicle, x0=x0, xf=xf, N=N, dt=dt, cones=cones, w_c=10000)

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(X[:, 0], X[:, 1], 'b-', label='Trajectory')
    plt.plot(x_guess[:,0], x_guess[:,1], '--', label='initial guess')
    plt.plot(x0[0],x0[1], 'x')
    plt.plot(xf[0],xf[1], 'x')
    for i, cone in enumerate(cones):
        color = 'r' if i % 2 == 0 else 'g'
        plt.plot(cone['x'], cone['y'], 'o', color=color, markersize=10)
        # plt.gca().add_patch(plt.Circle((cx, cy), cone_radius, color=color, fill=False))
    # plt.title(f'Slalom Trajectory (T = {T_opt:.2f}s)')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.grid()
    plt.legend()


    # Time vector
    t = np.linspace(0, N, N+1)

    # Plot states
    x_labels = ['x', 'y', 'theta', 'v']
    u_labels = ['a', 'omega']
    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(6, 1, i+1)
        plt.plot(t, X[:, i])
        plt.ylabel(x_labels[i])
        plt.grid(True)
    for i in range(2):
        plt.subplot(6, 1, i+5)
        plt.plot(t[:-1], U[:, i])
        plt.ylabel(u_labels[i])
        plt.grid(True)
    plt.tight_layout()
    plt.show()

def rrt_demo():
    dubins = setup_system()

    x0 = np.array([0]*5)
    xf = np.array([0,7,np.pi,0,0])
    rrt_settings = settings.get('demo', 'rrt')
    
    run_rrt(dubins, x0, xf, rrt_settings)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        demo = sys.argv[1]

    demos = {
        'optimization_demo': optimization_demo,
        'test_1': test_1,
        'rrt_demo': rrt_demo
    }
    if demo in demos:
        demos[demo]()
    else:
        print(f"Unknown demo: {demo}. Available demos: {', '.join(demos.keys())}")
        sys.exit(1)