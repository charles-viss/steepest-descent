import numpy as np


METHODS = {'auto': -1,
		   'primal_simplex': 0,
           'dual_simplex': 1,
           'barrier': 2,
           'concurrent': 3,
           'deterministic_concurrent': 4,
           'deterministic_concurrent_simplex': 5,}

INF = 10e100
EPS = 10e-10


class result:
    def __init__(self, status, x=None, obj=None, n_iters=None, solve_time=None, alg_type='simplex',
                 circuits=[], steps=[], simplex_iters=[], solve_times=[]):
        self.status = status
        self.x = x
        self.obj = obj
        self.n_iters = n_iters
        self.solve_time = solve_time
        self.alg_type = alg_type
        
        self.circuits = circuits
        self.steps = steps
        self.simplex_iters = simplex_iters
        self.solve_times = solve_times
          
    def __str__(self):
        if self.status == 1:
            return ('Problem is unbounded.'
                        + '\nSteepest descent unbounded circuit: ' + str(self.circuits[-1].T)
                    )
        elif self.status == 0:
            output = ('\nOptimal objective: {}'.format(self.obj)
                   + '\nTotal solve time: {}'.format(self.solve_time)
                   + '\nNumber of iterations: {}' + str(len(self.circuits))
                   + '\nFirst simplex iterations {}'.format(self.simplex_iters[0])
                   + '\nAverage num simplex iterations {}'.format(sum(self.simplex_iters)/len(self.simplex_iters))
                   + '\nTotal simplex iterations: {}'.format(sum(self.simplex_iters))
                   + '\nFirst solve time {}'.format(self.solve_times[0])
                   + '\nAverage solve time {}'.format(sum(self.solve_times)/len(self.solve_times))
                   + '\nTotal solve time: {}'.format(sum(self.solve_times))
            )
            if self.alg_type = 'steepest_descent':
                output += 
            return output