import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from dynamics.dubins import VehicleStateSpace, VehicleConfigurationSpace
import shapely


def trajectory_optimization(configurationspace : VehicleConfigurationSpace,
                            x0 : np.array, xf : np.array,
                            N : int,
                            dt : float,
                            x_guess=None,
                            u_guess=None):
    assert len(x0) == 5
    assert len(xf) == 5
    assert N >= 1
    assert dt > 0

    def sample_initial_guess():
        # x_guess = np.linspace(x0, xf, N+1)
        x_guess = np.zeros((N+1, len(x0)))

        u_guess = [None]*N
        for i in range(N):
            u_guess[i] = np.array([np.random.uniform(*configurationspace.dynamics.ax_bounds),
                                   np.random.uniform(*configurationspace.dynamics.psi_bounds)])
        return x_guess, u_guess
    
    if x_guess or u_guess is None:
        x_guess, u_guess = sample_initial_guess()

    # casadi function
    x = ca.SX.sym('x',len(x0))
    u = ca.SX.sym('u',len(u_guess[0]))

    dx = configurationspace.dynamics.x_dot(x, u, casadi=True)
    dynamic_constraints = [x + dt * dx]

    forward_dynamics = ca.Function('f', [x,u], dynamic_constraints)

    # optimization variables
    opti = ca.Opti()
    X = opti.variable(N+1, len(x0))
    U = opti.variable(N,len(u_guess[0]))

    def terminal_cost(x):
        return (x[0]-xf[0])**2+(x[1]-xf[1])**2+(x[3]-xf[3])**2
    
    def control_cost(u):
        return (u[0]**2 + u[1]**2)
    
    def state_cost(x):
        #obstacle avoidance costs
        d_safe = 0.5
        k = 1000
        cost = 0.0
        for obs in configurationspace.obstacles:
            center = obs.centroid
            boundary_point = list(obs.exterior.coords)[0]
            radius = center.distance(shapely.geometry.Point(boundary_point))
            d = ca.sqrt((x[0] - center.x)**2 + (x[1] - center.y)**2)
            cost += ca.if_else(
                d-radius < d_safe,
                k * (1/(d-radius) - 1/d_safe)**2,
                0.0
            )
        return cost

    def running_cost(x,u):
        return control_cost(u) + state_cost(x)
    
    #setup objective function
    objective_function = terminal_cost(X[-1,:])
    for i in range(N):
        objective_function = objective_function + dt * running_cost(X[i,:],U[i,:])

    opti.minimize(objective_function)

    # initial state
    opti.subject_to(X[0,:] == x0.reshape((1,5)))

    # dynamics bounds
    for i in range(N):
        opti.subject_to(X[i+1,:] == forward_dynamics(X[i,:],U[i,:]).T)

    # dynamic limit bounds
    for i in range(N):
        opti.subject_to(opti.bounded(configurationspace.dynamics.v_bounds[0],
                                     X[i+1,3],configurationspace.dynamics.v_bounds[1]))
        opti.subject_to(opti.bounded(configurationspace.dynamics.phi_bounds[0],
                                     X[i+1,4],configurationspace.dynamics.phi_bounds[1]))
        opti.subject_to(opti.bounded(configurationspace.dynamics.ax_bounds[0],
                                     U[i,0],configurationspace.dynamics.ax_bounds[1]))
        opti.subject_to(opti.bounded(configurationspace.dynamics.psi_bounds[0],
                                     U[i,1],configurationspace.dynamics.psi_bounds[1]))


    p_opts = {"expand":False}
    s_opts = {"max_iter": 100}
    opti.solver("ipopt",p_opts,
                        s_opts)
    
    for i  in range(N):
        opti.set_initial(X[i,:],x_guess[i])
        opti.set_initial(U[i,:],u_guess[i])

    max_attempts = 1
    solved = False
    best_solution = None
    best_cost = float('inf')

    for attempt in range(max_attempts):
        try:
            sol = opti.solve()
            solved = True

            current_cost = float(opti.value(opti.f))

            print(f"Success! Cost: {current_cost:.4f}")

            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = sol
        except RuntimeError as e:
            status = opti.stats()['return_status']
            if status == 'Maximum_Iterations_Exceeded':
                print(f"[Attempt {attempt+1}] Max iterations exceeded, retrying...")

                # Optional debug
                x_val = opti.debug.value(X)
                u_val = opti.debug.value(U)
                print(f"Debug X[0]: {x_val[0]}, U[0]: {u_val[0]}")

                x_guess, u_guess = sample_initial_guess()

                for i  in range(N):
                    opti.set_initial(X[i,:],x_guess[i])
                    opti.set_initial(U[i,:],u_guess[i])

            else:
                raise e
            
    if not solved:
        raise RuntimeError("Optimization failed after maximum retries.")

    print("Terminal cost",'%.3f'%terminal_cost(best_solution.value(X)[N,:]))

    return (best_solution.value(X),best_solution.value(U),best_solution.value(objective_function))
    

def cone_trajectory_optimization(vehicle : VehicleStateSpace,
                        x0 : np.array, xf : np.array,
                        N : int,
                        dt : float, cones=None, w_c=None):
    assert len(x0) == 4
    assert len(xf) == 4
    assert N >= 1
    assert dt > 0
    
    # casadi function
    x = ca.SX.sym('x',4)
    u = ca.SX.sym('u',2)

    dx = vehicle.x_dot(x, u)
    dynamic_constraints = [x + dt * dx]

    forward_dynamics = ca.Function('f', [x,u], dynamic_constraints)

    # Initial guess
    def initial_guess():
        offset = 2
        side = 1

        if cones is not None:
            segment = np.floor((N+1) / (len(cones) + 1)).astype(int)
            diff = (N+1) - segment * (len(cones) + 1)
            x_init =  np.linspace(x0[0], cones[0]['x'] + offset, segment+diff)
            y_init =  np.linspace(x0[1], cones[0]['y'], segment+diff)
            for i, cone in enumerate(cones[:-1]):
                x_init = np.concatenate((x_init,
                                        np.linspace(cone['x'] + side * offset,
                                                    cones[i+1]['x'] - side * offset,
                                                    segment)))
                y_init = np.concatenate((y_init, np.linspace(cone['y'], cones[i+1]['y'], segment)))
                side *= -1
            x_init = np.concatenate((x_init, np.linspace(x_init[-1], xf[0], segment)))
            y_init = np.concatenate((y_init, np.linspace(y_init[-1], xf[1], segment)))

            dx = np.diff(x_init)
            dy = np.diff(y_init)

            theta_init = np.arctan2(dx, dy)
            theta_init = np.append(theta_init, theta_init[-1])
        else:
            x_init = np.linspace(x0[0], xf[0], N+1)
            y_init = np.linspace(x0[1], xf[1], N+1)

            theta_init = [None]*(N+1)
            for i in range(N+1):
                theta_init[i] = np.random.uniform(*vehicle.theta_bounds)

        v_init = np.ones(N+1) * 5

        # u_guess = np.zeros((N, 2))
        u_guess = [None]*N
        for i in range(N):
            u_guess[i] = np.array([np.random.uniform(*vehicle.ax_bounds),np.random.uniform(*vehicle.omega_bounds)])
    
        x_guess = np.concatenate([x_init, y_init, theta_init, v_init])
        x_guess_sorted = np.stack((x_init, y_init, theta_init, v_init), axis=1)

        return x_guess, u_guess, x_guess_sorted

    x_guess, u_guess, x_guess_sorted = initial_guess()

    # optimization variables
    opti = ca.Opti()
    X = opti.variable(N+1, 4)
    U = opti.variable(N,2)

    def terminal_cost(x):
        return (x[0]-xf[0])**2+(x[1]-xf[1])**2
    
    def cone_cost(x):
        d_safe = 0.5
        cone_radius = 0.5
        d_near = 3
        cost = 0.0
        side = 1
        for cone in cones:
            cx, cy = cone['x'], cone['y']
            d = ca.sqrt((x[0] - cx)**2 + (x[1] - cy)**2)
            cost += ca.if_else(d - cone_radius <  d_safe,
                w_c * (1/(d - cone_radius + 1e-3) - 1/d_safe)**2,
                0.0
            )
        return cost

    def waypoint_generate(cones):
        slalom_waypoints = []
        side = 1
        lateral_offset = 1.0  # how far left/right to guide

        for cone in cones:
            wp = {
                'x': cone['x'] + side * lateral_offset,
                'y': cone['y']
            }
            slalom_waypoints.append(wp)
            side *= -1
        return slalom_waypoints

    def waypoint_cost(slalom_waypoints):
        segment = N // len(slalom_waypoints)
        cost = 0.0
        for i, wp in enumerate(slalom_waypoints):
            idx = min(i * segment + segment // 2, N-1)
            x = X[idx, :]

            # Soft penalty to drive trajectory near slalom waypoint
            d = ((x[0] - wp['x'])**2 + (x[1] - wp['y'])**2)
            cost += w_c * 1/d
        return cost


    #objective function
    objective_function = terminal_cost(X[-1,:])

    for i in range(N):
        objective_function = objective_function + cone_cost(X[i,:])

    slalom_waypoints = waypoint_generate(cones)
    objective_function += waypoint_cost(slalom_waypoints)

    # d_safe = 0.5
    # cone_radius = 0.5
    # d_near = 3
    # side = 1

    # segment = N // len(cones)

    # for i, cone in enumerate(cones):
    #     cx, cy = cone['x'], cone['y']
    #     cone_index = min(i * segment + segment // 2, N-1)  # pick a point near cone

    #     x = X[cone_index, :]
    #     d = ca.sqrt((x[0] - cx)**2 + (x[1] - cy)**2)

    #     # Distance penalty
    #     objective_function += ca.if_else(d - cone_radius < d_safe,
    #         w_c * (1/(d - cone_radius + 1e-3) - 1/d_safe)**2,
    #         0.0)

    #     # Side penalty
    #     side_offset = x[0] - cx
    #     # in_y_range = ca.logic_and(x[1] > cy - 1, x[1] < cy + 1)
        
    #     wrong_side = ca.logic_or(ca.logic_and(side > 0, side_offset <= 0),
    #                              ca.logic_and(side < 0, side_offset >= 0))
        
    #     objective_function += ca.if_else(wrong_side,
    #             w_c * (1 / (1e-3))**2,
    #             0.0
    #         )

    #     side *= -1



    opti.minimize(objective_function)

    # initial state
    opti.subject_to(X[0,:] == x0.reshape((1,4)))

    # dynamics bounds
    for i in range(N):
        opti.subject_to(X[i+1,:] == forward_dynamics(X[i,:],U[i,:]).T)

    # dynamic limit bounds
    for i in range(N):
        opti.subject_to(opti.bounded(vehicle.theta_bounds[0],X[i+1,2],vehicle.theta_bounds[1]))
        opti.subject_to(opti.bounded(vehicle.v_bounds[0],X[i+1,3],vehicle.v_bounds[1]))
        opti.subject_to(opti.bounded(vehicle.ax_bounds[0],U[i,0],vehicle.ax_bounds[1]))
        opti.subject_to(opti.bounded(vehicle.omega_bounds[0],U[i,1],vehicle.omega_bounds[1]))


    p_opts = {"expand":False}
    s_opts = {"max_iter": 500}
    opti.solver("ipopt",p_opts,
                        s_opts)
    
    for i  in range(N):
        opti.set_initial(X[i,:],x_guess[i])
        opti.set_initial(U[i,:],u_guess[i])

    # sol = opti.solve()
    max_attempts = 5
    solved = False
    best_solution = None
    best_cost = float('inf')

    for attempt in range(max_attempts):
        try:
            sol = opti.solve()
            solved = True

            current_cost = float(opti.value(opti.f))

            print(f"Success! Cost: {current_cost:.4f}")

            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = sol
        except RuntimeError as e:
            status = opti.stats()['return_status']
            if status == 'Maximum_Iterations_Exceeded':
                print(f"[Attempt {attempt+1}] Max iterations exceeded, retrying...")

                # Optional debug
                x_val = opti.debug.value(X)
                u_val = opti.debug.value(U)
                print(f"Debug X[0]: {x_val[0]}, U[0]: {u_val[0]}")

                x_guess, u_guess, x_guess_sorted = initial_guess()

                for i  in range(N):
                    opti.set_initial(X[i,:],x_guess[i])
                    opti.set_initial(U[i,:],u_guess[i])

                # Optional: adjust solver or initial guess
                # opti.set_initial(X, x_val)  # retry from current best guess
                # opti.set_initial(U, u_val)
            else:
                raise e
            
    if not solved:
        raise RuntimeError("Optimization failed after maximum retries.")

    print("Terminal cost",'%.3f'%terminal_cost(best_solution.value(X)[N,:]))
    print("Cone cost",'%.3f'%cone_cost(best_solution.value(X)[N,:]))

    return (best_solution.value(X),best_solution.value(U),best_solution.value(objective_function), x_guess_sorted)

    



