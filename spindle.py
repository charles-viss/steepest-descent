import numpy as np
from utils import EPS

from polyhedron import Polyhedron

# constructs a bounded/fixed-size partition polytope with the given parameters
# n is number of items,  k is the number of clusters,
# ub are cluster size upper bounds, and lb are cluster size lower bounds
class Spindle(Polyhedron):
    def __init__(self, n, n_cone_facets, n_parallel_facets, c=None):
        
        self.n = n
        self.n_cone_facets = n_cone_facets
        self.n_parallel_facets = n_parallel_facets
        self.B = []
        self.d = []

        # two points in R^n used to define the spindle
        self.p1 = np.zeros(n)
        self.p2 = np.ones(n)
        self.c = self.p1 - self.p2

        # add facets containing p1 that do not contain p2
        for _ in range(self.n_cone_facets):
            row = np.random.randint(-100, 25, size=self.n)
            rhs = row.dot(self.p1)
            while row.dot(self.p2) >= rhs:
                row = np.random.randint(-100, 25, size=self.n)   
                rhs = row.dot(self.p1)
            self.B.append(row)
            self.d.append(rhs)
            assert row.dot(self.p2) < rhs
            
        # add facets containing p2 that do not contain p1
        for _ in range(self.n_cone_facets):
            row = np.random.randint(-25, 100, size=self.n)
            rhs = row.dot(self.p2)
            while row.dot(self.p1) >= rhs:
                row = np.random.randint(-25, 100, size=self.n)   
                rhs = row.dot(self.p2)
            self.B.append(row)
            self.d.append(rhs)
            assert row.dot(self.p1) < rhs
            
        # add parallel facets that don't contain p1 or p2 but contain other points of the unit cube
        for _ in range(self.n_parallel_facets):
            row = np.random.randint(-10, 10, size=self.n)
            point = np.random.randint(0, 2, size=self.n)
            rhs = row.dot(point)
            while row.dot(self.p1) >= rhs or row.dot(self.p2) >= rhs:
                row = np.random.randint(-10, 10, size=self.n)
                point = np.random.randint(0, 2, size=self.n)
                rhs = row.dot(point)
            self.B.append(row)
            self.d.append(rhs)
            row2 = -1*row
            rhs2 = row2.dot(np.ones(self.n) - point)
            assert row2.dot(self.p1) < rhs2 and row2.dot(self.p2) < rhs2
            self.B.append(row2)
            self.d.append(rhs2)      

        self.B = np.asarray(self.B, dtype=np.int16)
        self.d = np.asarray(self.d, dtype=np.int16)        
        super(Spindle, self).__init__(self.B, self.d, A=None, b=None, c=self.c)

    def find_feasible_solution(self, verbose=False):
        self.x_current = self.p1
        self.B_x_current = self.B.dot(self.x_current)
        return self.x_current
    
    def get_active_constraints(self, x_current=None):
        if x_current is not None: self.x_current = x_current
        if np.array_equal(self.x_current, self.p1):
            inds = list(range(self.n_cone_facets))
            self.active_inds = [False]*self.m_B
            for i in inds:
                self.active_inds[i] = True
            return inds
        else:
            return super(Spindle, self).get_active_constraints(x_current)