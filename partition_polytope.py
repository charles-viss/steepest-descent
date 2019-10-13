import numpy as np
from utils import EPS

from polyhedron import Polyhedron

# constructs a bounded/fixed-size partition polytope with the given parameters
# n is number of items,  k is the number of clusters,
# ub are cluster size upper bounds, and lb are cluster size lower bounds
class PartitionPolytope(Polyhedron):
    def __init__(self, n_items, k, ub, lb, c=None):
        assert len(ub) == k and len(lb) == k, 'Invalid cluster size bounds'
        self.n_items = n_items
        self.k = k
        self.ub = ub
        self.lb = lb

        self.A = []
        self.b = []
        self.B = []
        self.d = []
        self.c = c

        # unique item assignment constraints
        for j in range(n_items):
            row = np.zeros(n_items*k, dtype=np.int16)
            for i in range(k):
                row[i*n_items + j] = 1
            self.A.append(row)
            self.b.append(1)

        # cluster size constraints
        self.fixed_cluster_inds = []
        self.bounded_cluster_inds = []
        for i in range(k):
            row = np.zeros(n_items*k, dtype=np.int16)
            for j in range(n_items):
                row[i*n_items + j] = 1
            if ub[i] == lb[i]:
                self.A.append(row)
                self.b.append(ub[i])
                self.fixed_cluster_inds.append(i)
            elif ub[i] > lb[i]:
                self.B.append(row)
                self.B.append(-1*row)
                self.d.append(ub[i])
                self.d.append(-1*lb[i])
                self.bounded_cluster_inds.append(i)
            else:
                raise ValueError('Invalid cluster size bounds')
        self.n_fixed_clusters = len(self.fixed_cluster_inds)
        self.n_bounded_clusters = len(self.bounded_cluster_inds)

        # variable nonnegativity constraints
        for i in range(n_items*k):
            row = np.zeros(n_items*k, dtype=np.int16)
            row[i] = -1
            self.B.append(row)
            self.d.append(0)

        self.A = np.asarray(self.A, dtype=np.int16)
        self.b = np.asarray(self.b, dtype=np.int16)
        self.B = np.asarray(self.B, dtype=np.int16)
        self.d = np.asarray(self.d, dtype=np.int16)        
        super(PartitionPolytope, self).__init__(self.B, self.d, self.A, self.b, self.c)

    def get_constraint_matrices(self):
        return (self.A, self.b, self.B, self.d)
    
    def find_feasible_solution(self, verbose=False):
        y = np.zeros(self.n_items*self.k, dtype=np.int16)
        self.cluster_sizes = np.zeros(self.k, dtype=np.int16)
        items_assigned = 0
        for i in range(self.k):
            for _ in range(self.lb[i]):
                y[i*self.n_items + items_assigned] = 1
                items_assigned += 1
                self.cluster_sizes[i] += 1
        
        i = 0
        n_loops = 0
        while items_assigned < self.n_items:
            if self.cluster_sizes[i] < self.ub[i]:
                y[i*self.n_items + items_assigned] = 1
                items_assigned += 1
                self.cluster_sizes[i] += 1
            i = (i + 1) % self.k
            n_loops += 1
            if n_loops == self.k * self.n_items:
                raise RuntimeError('Unable to find feasible clustering')
                           
        assert np.sum(y) == self.n_items
        self.y_current = y
        return y
    
    def get_active_constraints(self, y_current=None):
        if y_current is not None: self.y_current = y_current
        inds = []
        for j, i in enumerate(self.bounded_cluster_inds):
            if self.cluster_sizes[i] == self.ub[i]:
                inds.append(2*j)
            elif self.cluster_sizes[i] == self.lb[i]:
                inds.append(2*j + 1)
        n_cluster_ineqs = 2*(self.n_bounded_clusters)
        for j in range(self.n_items*self.k):
            if self.y_current[j] == 0:
                inds.append(n_cluster_ineqs + j)
        self.active_inds = [False]*self.m_B
        for i in inds:
            self.active_inds[i] = True
        return inds
    
    def take_maximal_step(self, g, y_pos, y_neg):
        assert hasattr(self, 'y_current') and hasattr(self, 'active_inds') 
        
        # normalize g to be a 0/1-vector
        scale = 0
        for g_i in g:
            if g_i > 0:
                scale = (1./g_i)
                break
        g_normalized = np.round(scale*g).astype(np.int16)
        
        # take step with size 1
        alpha = 1
        self.y_current += g_normalized
        assert np.sum(self.y_current) == self.n_items

        for j, i in enumerate(self.bounded_cluster_inds):
            if y_pos[2*j] > EPS:
                self.cluster_sizes[i] += 1
                self.active_inds[2*j + 1] = False
                if self.cluster_sizes[i] == self.ub[i]:
                    self.active_inds[2*j] = True
                elif self.cluster_sizes[i] > self.ub[i]:
                    raise RuntimeError('Invalid step')
            elif y_neg[2*j] > EPS:
                self.cluster_sizes[i] -= 1
                self.active_inds[2*j] = False
                if self.cluster_sizes[i] == self.lb[i]:
                    self.active_inds[2*j + 1] = True
                elif self.cluster_sizes[i] < self.lb[i]:
                    raise RuntimeError('Invalid step')
        
        n_cluster_ineqs = 2*(self.n_bounded_clusters)
        for j in range(self.n_items*self.k):
            if self.y_current[j] == 0:
                self.active_inds[n_cluster_ineqs + j] = True
            elif self.y_current[j] == 1:
                self.active_inds[n_cluster_ineqs + j] = False
                
        return self.y_current, alpha, [i for i in range(self.m_B) if self.active_inds[i]]

