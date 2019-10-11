import pickle
import random
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

def avg(x):
    return float(sum(x)) / float(len(x))


# constructs a bounded/fixed-size partition polytope with the given parameters
# n is number of items,  k is the number of clusters,
# ub are cluster size upper bounds, and lb are cluster size lower bounds
class PartitionPolytope():
    def __init__(self, n, k, ub, lb):
        assert len(ub) == k and len(lb) == k, 'Invalid cluster size bounds'
        self.n = n
        self.k = k
        self.ub = ub
        self.lb = lb

        self.A = []
        self.b = []
        self.B = []
        self.d = []

        # unique item assignment constraints
        for j in range(n):
            row = np.zeros(n, dtype=np.uint8)
            for i in range(k):
                row[i*n + j] = 1
            self.A.append(row)
            self.b.append(1)

        # cluster size constraints
        for i in range(k):
            row = np.zeros(n, dtype=np.uint8)
            for j in range(n):
                row[i*n + j] = 1
            if ub[i] == lb[i]:
                self.A.append(row)
                self.b.append(ub[i])
            elif ub[i] > lb[i]:
                self.B.append(row)
                self.B.append(-1*row)
                self.d.append(ub[i])
                self.d.append(-1*lb[i])
            else:
                raise ValueError('Invalid cluster size bounds')

        # variable nonnegativity constraints
        for i in range(n*k):
            row = np.zeros(n, dtype=np.uint8)
            row[i] = -1
            self.B.append(row)
            self.d.append(0)

        self.A = np.asarray(self.A, dtype=np.uint8)
        self.b = np.asarray(self.b, dtype=np.uint8)
        self.B = np.asarray(self.B, dtype=np.uint8)
        self.d = np.asarray(self.d, dtype=np.uint8)

    def get_constraint_matrices(self):
        return (self.A, self.b, self.B, self.d))


class result:
    
    def __init__(self, status, x=None, obj=None, n_iters=None, solve_time=None, alg_type='simplex',
                 circuits=[], steps=[], simplex_iters=[], solve_times=[], sub_times=None,
                 obj_values=[]):
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
        
        self.sub_times = sub_times
        self.obj_values = obj_values
          
    def __str__(self):
        if self.status == 1:
            return ('Problem is unbounded.'
                        + '\nSteepest descent unbounded circuit: ' + str(self.circuits[-1].T)
                    )
        elif self.status == 0:
            output = ('\nOptimal objective: {}'.format(self.obj)
                   + '\nTotal solve time: {}'.format(self.solve_time)
                   + '\nNumber of iterations: {}'.format(self.n_iters)
            )
            if self.alg_type == 'steepest-descent':
                output += ('\nFirst simplex iterations {}'.format(self.simplex_iters[0])
                       + '\nAverage num simplex iterations {}'.format(sum(self.simplex_iters)/len(self.simplex_iters))
                       + '\nTotal simplex iterations: {}'.format(sum(self.simplex_iters))
                       + '\nFirst solve time {}'.format(self.solve_times[0])
                       + '\nAverage solve time {}'.format(sum(self.solve_times)/len(self.solve_times))
                       + '\nTotal solve time: {}'.format(sum(self.solve_times))
                       )
            return output
        else:
            return 'Problem unsolved'
        
    def save(self, fn):
        results = {'obj': self.obj, 
                   'obj_values': self.obj_values,
                   'n_iters': self.n_iters, 
                   'solve_time_total': self.solve_time,
                   'alg_type': self.alg_type}
        if self.alg_type == 'steepest-descent':
            results['simplex_iters'] = self.simplex_iters
            results['solve_times'] = self.solve_times
            results['sub_times'] = self.sub_times
            
        with open(fn, 'wb') as f:
            pickle.dump(results, f)
