from mps_reader_preprocessor import read_mps_preprocess
from polyhedron import Polyhedron
from steepest_descent import steepest_descent_augmentation_scheme as sdac
import time
import os
import random

test_dir = 'test_problems'
mps_fns = os.listdir(test_dir)
mps_fn = random.choice(mps_fns)
mps_fn = os.path.join(test_dir, mps_fn)

#mps_fn = 'brandy.mps' # works
#mps_fn = 'lotfi.mps' # works
#mps_fn = 'agg3.mps'
#mps_fn = 'pilot.mps'
#mps_fn = 'test_problems/ship12l' # bad case
mps_fn = 'test_problems/standgub' # good example

print('Reading {}...'.format(mps_fn))
c, B, d, A, b = read_mps_preprocess(mps_fn)

print('Building polyhedron...')
P = Polyhedron(B, d, A, b, c)

print('Finding feasible solution...')
x_feasible, model, x_var = P.find_feasible_solution(verbose=False)

print('\nSolving with simplex method...')
x_optimal, obj_optimal, num_steps, solve_time = P.solve_lp(model=model, x_var=x_var, verbose=False)
print('\nSolution using simplex method:')
print('Objective value: {}'.format(obj_optimal))
print('Number of iterations: {}'.format(num_steps))
print('Solve time: {}'.format(solve_time))

time.sleep(3)

print('\nSolving with steepest descent...')
result = sdac(P, c, x_feasible)
x_optimal = result.x
print('\nSolution using steepest-descent augmentation:')
print(result)
#for i in range(len(result.circuits)):
#    print('g_' + str(i) + ': ' + str(result.circuits[i].T) + ' with alpha_'
#              + str(i) + ' = ' + str(result.steps[i]))