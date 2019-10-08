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

#mps_fn = 'agg3.mps'
#mps_fn = 'pilot.mps'
#mps_fn = 'test_problems/ship12l' # long example

#mps_fn = 'test_problems/standgub' # fast examples
mps_fn = 'test_problems/finnis'
#mps_fn = 'test_problems/degen2'
#mps_fn = 'test_problems/brandy.mps'
#mps_fn = 'test_problems/lotfi.mps'

#mps_fn = 'test_problems/qap08 # long but successful 


print('Reading {}...'.format(mps_fn))
c, B, d, A, b = read_mps_preprocess(mps_fn)

print('Building polyhedron...')
P = Polyhedron(B, d, A, b, c)

print('Finding feasible solution...')
x_feasible = P.find_feasible_solution(verbose=False)

print('\nSolving with simplex method...')
lp_result = P.solve_lp(verbose=False, record_objs=True)
print('\nSolution using simplex method:')
print(lp_result)

time.sleep(2)

print('\nSolving with steepest descent...')
sd_result = sdac(P, x_feasible, c=c, method='dual_simplex')
x_optimal = sd_result.x
print('\nSolution for {} using steepest-descent augmentation:'.format(os.path.basename(mps_fn)))
print(sd_result)
#for i in range(len(result.circuits)):
#    print('g_' + str(i) + ': ' + str(result.circuits[i].T) + ' with alpha_'
#              + str(i) + ' = ' + str(result.steps[i]))

print('\n\nSolution for {} using simplex method:'.format(os.path.basename(mps_fn)))
print(lp_result)