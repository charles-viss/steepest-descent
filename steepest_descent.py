import math
import time

from utils import result, EPS

def steepest_descent_augmentation_scheme(P, x, c=None, verbose=False, method='dual_simplex',
                                         max_time=300):
    """
Given a polyhedron P with feasible point x and an objective function c,
solve the linear program min{c^T x : x in P} via the steepest descent circuit augmentation scheme.
Returns result object containing optimal solution, objective objective, solve time, and other stats.
    """
    
    if c is not None:
        P.set_objective(c)
 
    t0 = time.time()
    x_current = x
    active_inds = P.get_active_constraints(x_current)
    pm = P.build_polyhedral_model(active_inds=active_inds, method=method)
    t1 = time.time()
    build_time = t1 - t0
    print('Polyhedral model build time: {}'.format(build_time))
    
    sub_times = {'sd': [], 'step': [], 'solve': []}    
    descent_circuits = []
    obj_values = []
    step_sizes = []
    iter_times = []
    simplex_iters = []
    iteration = 0
    t2 = time.time()   
    
    # compute steepest-descent direction
    descent_direction, y_pos, y_neg, steepness, num_steps, solve_time = pm.compute_sd_direction(verbose=verbose)
    simplex_iters.append(num_steps)
    sub_times['solve'].append(solve_time)
    
    t3 = time.time()
    sub_times['sd'].append(t3 - t2)
    
    while abs(steepness) > EPS:
        
        t3 = time.time()
        obj_values.append(P.c.dot(x_current))
        iter_times.append(t3 - t1)
        
        # take maximal step
        x_current, alpha, active_inds = P.take_maximal_step(descent_direction, y_pos, y_neg)  
        
        t4 = time.time()
        sub_times['step'].append(t4 - t3) 
        descent_circuits.append(descent_direction)
        step_sizes.append(alpha)     
                
        if math.isinf(alpha):
            # problem is unbounded
            return result(status=1, circuits=descent_circuits, steps=step_sizes)
        
        # compute steepest-descent direction
        pm.set_active_inds(active_inds)
        descent_direction, y_pos, y_neg, steepness, num_steps, solve_time = pm.compute_sd_direction(verbose=verbose)
        
        t5 = time.time()
        sub_times['sd'].append(t5 - t4)
        sub_times['solve'].append(solve_time)
        simplex_iters.append(num_steps)
        
        iteration += 1
        current_time = t5 - t1
        if current_time > max_time:
            return result(status=2)

    t6 = time.time()
    total_time = t6 - t1   
    print('Total time for steepest-descent scheme: {}'.format(total_time))
        
    return result(status=0, x=x_current, 
                  obj=P.c.dot(x_current), n_iters=len(step_sizes), solve_time=total_time,
                  iter_times=iter_times, alg_type='steepest-descent',
                  circuits=descent_circuits, steps=step_sizes,
                  simplex_iters=simplex_iters, solve_times=sub_times['solve'],
                  sub_times=sub_times, obj_values=obj_values)
