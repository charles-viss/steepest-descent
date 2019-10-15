from mps_reader_preprocessor import read_mps_preprocess
from polyhedron import Polyhedron
from steepest_descent import steepest_descent_augmentation_scheme as sdac
import time
import random
import os
import random
import numpy as np

from partition_polytope import PartitionPolytope
from spindle import Spindle


def main(mps_fn='', results_dir=None, max_time=300, sd_method='dual_simplex',
         partition_polytope=False, n=0, k=0,
         spindle=False, spindle_dim=0, n_cone_facets=0, n_parallel_facets=0):
    
    if mps_fn:
        print('Reading {}...'.format(mps_fn))
        c, B, d, A, b = read_mps_preprocess(mps_fn)
        print('Building polyhedron...')
        P = Polyhedron(B, d, A, b, c)
    elif partition_polytope:
        print('Constructing partition polytope with n={} and k={}'.format(n, k))
        # randomly generate cluster size bounds and objective function
        v1 = np.random.randint(0, n, size=k)
        v2 = np.random.randint(0, n//k, size=k)
        ub = [max(v1[i], v2[i]) for i in range(k)]
        lb = [min(v1[i], v2[i]) for i in range(k)]
        c = np.random.randint(0, 1000, size=n*k)
        P = PartitionPolytope(n, k, ub, lb, c)
    elif spindle:
        print('Constructing spindle with dimension n={}, with {} cone facets,'
              'and with {} pairs of parallel facets'.format(
              spindle_dim, n_cone_facets, n_parallel_facets))
        P = Spindle(spindle_dim, n_cone_facets, n_parallel_facets)
        c = P.c
    else:
        raise RuntimeError('Provide argument for constructing polyhedron.')
    
    print('Finding feasible solution...')
    x_feasible = P.find_feasible_solution(verbose=False)
    if partition_polytope or spindle:
        print('Building gurobi model for simplex...')
        P.build_gurobi_model(c=c)
        P.set_solution(x_feasible)
    
    print('\nSolving with simplex method...')
    lp_result = P.solve_lp(verbose=False, record_objs=True)
    print('\nSolution using simplex method:')
    print(lp_result)
    
    print('\nSolving with steepest descent...')
    sd_result = sdac(P, x_feasible, c=c, method=sd_method, max_time=max_time)
    print('\nSolution for {} using steepest-descent augmentation:'.format(os.path.basename(mps_fn)))
    print(sd_result)
    
    print('\n\nSolution for {} using simplex method:'.format(os.path.basename(mps_fn)))
    print(lp_result)
    
    if results_dir:
        if not os.path.exists(results_dir): os.mkdir(results_dir)
        if mps_fn:
            prefix = os.path.basename(mps_fn).split('.')[0]
        elif partition_polytope: 
            prefix = 'n-{}_k-{}'.format(n, k)
        elif spindle:
            prefix = 'n-{}_c-{}_p-{}'.format(spindle_dim, n_cone_facets, n_parallel_facets)
        lp_fn = os.path.join(results_dir, prefix + '_lp.p')
        sd_fn = os.path.join(results_dir, prefix + '_sd.p')
        lp_result.save(lp_fn)
        sd_result.save(sd_fn)        


if __name__ == "__main__":
    
    mps_fns = os.listdir(problem_dir)
    mps_fn = random.choice(mps_fns)
    
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    #parser.add_argument('--mps_fn', help='mps filename for problem to solve', default=mps_fn)
    parser.add_argument('--mps_fn', help='mps filename for problem to solve', default='')
    parser.add_argument('--sd_method', help='algorithm for computing s.d. directions', type=str, default='dual_simplex')
    
    parser.add_argument('--partition-polytope', help='use bounded/fixed-size partition polytope', action='store_true')
    parser.add_argument('--n', help='num items for partition polytope', type=int, default=0)
    parser.add_argument('--k', help='num clusters for partition polytope', type=int, default=0)
    
    parser.add_argument('--spindle', help='use a spindle', action='store_true')
    parser.add_argument('--spindle_dim', help='dimension of spindle', type=int, default=0)
    parser.add_argument('--n_cone_facets', help='number of facets per cone of the spindle', type=int, default=0)
    parser.add_argument('--n_parallel_facets', help='number of pairs of parallel facet in spindle', type=int, default=0)
    
    parser.add_argument('--results_dir', help='directory for saving results', default='results')
    parser.add_argument('--max_time', help='max time for steepest descent algorithm in seconds',
                        default=300)

    args = parser.parse_args()
    
    main(mps_fn=args.mps_fn, results_dir=args.results_dir, max_time=args.max_time, sd_method=args.sd_method,
         partition_polytope=args.partition_polytope, n=args.n, k=args.k)