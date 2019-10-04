import math
from polyhedral_model import PolyhedralModel
import time

from utils import result

def steepest_descent_augmentation_scheme(P, x, c=None, verbose=False, method='dual_simplex'):
    """
Given a polyhedron P with feasible point x and an objective function c,
solve the linear program min{c^T x : x in P} via the steepest descent circuit augmentation scheme.
Returns result object containing optimal solution and objective function value.
    """
    
    if c is not None:
        P.set_objective(c)
    pm = P.build_polyhedral_model(x=x, method=method)
    
    descent_circuits = []
    step_sizes = []
    solve_times = []
    simplex_iters = []
    
    x_current = x
    active_inds = P.get_active_constraints(x_current)  
    pm.set_active_inds(active_inds)
    
    descent_direction, steepness, num_steps, solve_time = pm.compute_sd_direction(verbose=verbose)
    simplex_iters.append(num_steps)
    solve_times.append(solve_time)
    iteration = 0
    
    
    while steepness != 0:
        alpha, stopping_ind = P.get_max_step_size(x=x_current, g=descent_direction, 
                                                  active_inds=active_inds)
        
        if iteration % 20 == 0:
            print('\nIteration: {}'.format(len(step_sizes)))
            print('alpha: {}'.format(alpha))
            print('steepness: {}'.format(steepness))    
            #print('solve time: {}'.format(solve_time))
            #print('x_current:')
            #print(x_current)
            #time.sleep(0.1)
        
        descent_circuits.append(descent_direction)
        step_sizes.append(alpha)
        
        if math.isinf(alpha):
            return result(status=1, circuits=descent_circuits, steps=step_sizes)
        
        x_current = x_current + alpha*descent_direction
        active_inds = P.get_active_constraints(x_current)
        pm.set_active_inds(active_inds)
        descent_direction, steepness, num_steps, solve_time = pm.compute_sd_direction(verbose=verbose)
        simplex_iters.append(num_steps)
        solve_times.append(solve_time)
        
        iteration += 1
        
        
        #active_inds = P.get_active_constraints(x_current)
        #print('Active inds: {}'.format(active_inds))
        
    return result(status=0, x=x_current, circuits=descent_circuits, steps=step_sizes, c=c,
                  simplex_iters=simplex_iters, solve_times=solve_times)
