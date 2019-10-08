import math
import time

from utils import result, EPS

def steepest_descent_augmentation_scheme(P, x, c=None, verbose=False, method='dual_simplex'):
    """
Given a polyhedron P with feasible point x and an objective function c,
solve the linear program min{c^T x : x in P} via the steepest descent circuit augmentation scheme.
Returns result object containing optimal solution and objective function value.
    """
    
    if c is not None:
        P.set_objective(c)
 
    t0 = time.time()
    pm = P.build_polyhedral_model(x=x, method=method)
    t1 = time.time()

    build_time = t1 - t0
    print('Polyhedral model build time: {}'.format(build_time))
    sub_times = {'sd': [], 'active_inds': [], 'alpha': [], 'step': [], 'solve': []}
    
    descent_circuits = []
    obj_values = []
    step_sizes = []
    simplex_iters = []
    
    x_current = x
    active_inds = P.get_active_constraints(x_current)
    pm.set_active_inds(active_inds)
    
    t2 = time.time()
    sub_times['active_inds'].append(t2 - t1)
    
    descent_direction, y_pos, steepness, num_steps, solve_time = pm.compute_sd_direction(verbose=verbose)
    simplex_iters.append(num_steps)
    sub_times['solve'].append(solve_time)
    iteration = 0
    
    t3 = time.time()
    sub_times['sd'].append(t3 - t2)
    
    while abs(steepness) > EPS:

        obj_values.append(P.c.dot(x_current))
        t3 = time.time()
        alpha, stopping_ind = P.get_max_step_size(x=x_current, g=descent_direction, 
                                                  active_inds=active_inds, y_pos=y_pos)
        t4 = time.time()
        sub_times['alpha'].append(t4 - t3)

        if iteration % 20 == 0:
            print('\nIteration: {}'.format(len(step_sizes)))
            print('alpha: {}'.format(alpha))
            print('steepness: {}'.format(steepness))    
            #print('solve time: {}'.format(solve_time))
            #time.sleep(0.1)
        
        descent_circuits.append(descent_direction)
        step_sizes.append(alpha)
        
        if math.isinf(alpha):
            return result(status=1, circuits=descent_circuits, steps=step_sizes)
        
        x_current = x_current + alpha*descent_direction
        t5 = time.time()
        sub_times['step'].append(t5 - t4)

        active_inds = P.get_active_constraints(x_current)
        pm.set_active_inds(active_inds)
        t6 = time.time()
        sub_times['active_inds'].append(t6 - t5)

        descent_direction, y_pos, steepness, num_steps, solve_time = pm.compute_sd_direction(verbose=verbose)
        simplex_iters.append(num_steps)
        sub_times['solve'].append(solve_time)
        t7 = time.time()
        sub_times['sd'].append(t7 - t6)
        
        iteration += 1

    t8 = time.time()
    total_time = t8 - t1
    
    print('Total time for steepest-descent scheme: {}'.format(total_time))
        
    return result(status=0, x=x_current, 
                  obj=P.c.dot(x_current), n_iters=len(step_sizes), solve_time=total_time,
                  alg_type='steepest-descent',
                  circuits=descent_circuits, steps=step_sizes,
                  simplex_iters=simplex_iters, solve_times=sub_times['solve'],
                  sub_times=sub_times, obj_values=obj_values)
