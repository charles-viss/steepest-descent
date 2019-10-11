from mps_reader_preprocessor import read_mps_preprocess
from polyhedron import Polyhedron
from steepest_descent import steepest_descent_augmentation_scheme as sdac
import time
import os
import random



def main(mps_fn, results_dir=None, max_time=300):
    
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
    
    #time.sleep(2)
    
    print('\nSolving with steepest descent...')
    sd_result = sdac(P, x_feasible, c=c, method='dual_simplex', max_time=max_time)
    print('\nSolution for {} using steepest-descent augmentation:'.format(os.path.basename(mps_fn)))
    print(sd_result)
    
    print('\n\nSolution for {} using simplex method:'.format(os.path.basename(mps_fn)))
    print(lp_result)
    
    if results_dir:
        if not os.path.exists(results_dir): os.mkdir(results_dir)
        prefix = os.path.basename(mps_fn).split('.')[0]
        lp_fn = os.path.join(results_dir, prefix + '_lp.p')
        sd_fn = os.path.join(results_dir, prefix + '_sd.p')
        lp_result.save(lp_fn)
        sd_result.save(sd_fn)        


if __name__ == "__main__":
    
    mps_fns = os.listdir(problem_dir)
    mps_fn = random.choice(mps_fns)
    #mps_fn = os.path.join(test_dir, mps_fn)

    #mps_fn = 'agg3.mps'
    #mps_fn = 'pilot.mps'
    #mps_fn = 'test_problems/ship12l' # long example

    mps_fn = 'standgub' # fast examples
    #mps_fn = 'finnis'
    #mps_fn = 'degen2'
    #mps_fn = 'brandy.mps'
    
    #mps_fn = 'qap08 # long but successful 
    #mps_fn = pilotnov
    
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--mps_fn', help='mps filename for problem to solve', default=mps_fn)
    parser.add_argument('--results_dir', help='directory for saving results', default='results')
    parser.add_argument('--max_time', help='max time for steepest descent algorithm in seconds',
                        default=300)

    args = parser.parse_args()
    
    main(mps_fn=args.mps_fn, results_dir=args.results_dir, max_time=args.max_time)