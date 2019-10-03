import numpy as np
import sympy
import gurobipy as gp
#import contextlib

from polyhedral_model import PolyhedralModel

INF = 10e100
EPS = 10e-20

#class for representing a general polyhedron of the form:
# P = {x in R^n : Ax = b, Bx <= d}, with optional objective c
class Polyhedron:
    
    #initiallize with matrices and vectors given by numpy arrays
    def __init__(self, B, d, A=None, b=None, c=None):
        self.B = B
        self.d = d
        self.A = A
        self.b = b
        self.c = c
        
        self.m_B, self.n = self.B.shape
        self.m_A = self.A.shape[0] if self.A is not None else 0
    
    # get active constraints at given solution
    def get_active_constraints(self, x):
        B_x = self.B.dot(x)
        inds = []
        for i in range(self.m_B):
            if abs(self.d[i] - B_x[i]) <= 10e-10:
                inds.append(i)
        #print('Active inds:')
        #print(inds)        
        #print('Differences:')
        #for i in inds:
        #    print(self.d[i] - B_x[i])
        return inds   
    
    
    #given a point x in P with feasible direction g, compute the maximum step size alpha
    def get_max_step_size(self, x, g, active_inds=None):
        B_g = self.B.dot(g)     
        B_x = self.B.dot(x)
        alpha = float('inf')
        active_ind = None
        
        if active_inds is not None:
            inds = [i for i in range(self.m_B) if i not in active_inds]
        else:
            inds = range(self.m_B)            
        
        for i in inds:
            if B_g[i] > EPS:
                a = (self.d[i] - B_x[i]) / float(B_g[i])
                if a <= alpha:
                    alpha = a 
                    active_ind = i
        return alpha, active_ind
        
    # return normalized circuit given a circuit direction of P
    def get_normalized_circuit(self, g):
        B_g = self.B.dot(g)
        B_0 = np.zeros((1, self.n), dtype=int)
        
        for i in range(self.m_B):
            if abs(B_g[i]) <= EPS:
                B_0 = np.concatenate((B_0, self.B[i,:].reshape((1 ,self.n))))
        if self.A is not None:
                B_0 = np.concatenate((self.A, B_0), axis=0)
                
        D = sympy.Matrix(B_0)
        ker_D = D.nullspace()
        #if len(ker_D) != 1:
        #    raise ValueError('The direction ' + str(g.T)  +' is not a circuit of P')
        circuit = np.array(ker_D[0]).reshape(self.n)
        circuit= circuit*sympy.lcm([circuit[i].q for i in range(self.n) if circuit[i] != 0]) #normalize
        
        #make sure circuit has correct sign
        for i in range(self.n):
            if abs(g[i]) >= EPS:
                if circuit[i]*g[i] < 0:
                    circuit = -1*circuit
                break
        return circuit
    
    # construct polyhedral model for computing circuits
    def build_polyhedral_model(self, x=None, primal=True, method='dual_simplex'):
        print('Building polyhedral model...')
        active_inds = []
        if x is not None:
            active_inds = self.get_active_constraints(x)
        pm = PolyhedralModel(B=self.B, A=self.A, c=self.c, active_inds=active_inds)
        print('Polyhedral model built!')
        return pm
    
    # sovle linear program using traditional simplex method; option to give warm started model
    def solve_lp(self, c=None, model=None, x_var=None, verbose=False):
        
        if c is None:
            assert self.c is not None, 'Need objective function'
            c = self.c
        else:
            self.c = c
         
        if model is None or x_var is None:
            model, x = self.build_gurobi_model(c, verbose)
        else:
            x = x_var
       
        model.setObjective(gp.LinExpr(c, x))
        model.optimize()
        if model.status != gp.GRB.Status.OPTIMAL:
            raise RuntimeError('Model failed to solve')
            
        x_optimal = model.getAttr('x', x)
        obj_optimal = model.objVal
        num_steps = model.getAttr('IterCount')
        solve_time = model.getAttr('Runtime')
        
        return x_optimal, obj_optimal, num_steps, solve_time
    
    
    # find a feasible solution within the polyhedron
    def find_feasible_solution(self, verbose=False):
        
        c = np.zeros(self.n)            
        model, x = self.build_gurobi_model(c, verbose)
        
        model.optimize()
        if model.status != gp.GRB.Status.OPTIMAL:
            raise RuntimeError('Model failed to solve')
            
        x_feasible = model.getAttr('x', x)
        x_feasible = np.array(x_feasible)
        x_var = x
        return x_feasible, model, x_var

    # build a gurobi LP with the given objective            
    def build_gurobi_model(self, c, verbose=False):
    
        model = gp.Model()
        x = []
        for i in range(self.n):
            x.append(model.addVar(lb=-INF, ub=INF, name='x_{}'.format(i)))
        for i in range(self.m_A):
            model.addConstr(gp.LinExpr(self.A[i], x) == self.b[i], name='A_{}'.format(i))
        for i in range(self.m_B):
            model.addConstr(gp.LinExpr(self.B[i], x) <= self.d[i], name='B_{}'.format(i))
        model.setObjective(gp.LinExpr(c, x))
        
        #with contextlib.redirect_stdout(None):
        model.Params.method = 1
        flag = 1 if verbose else 0
        model.setParam(gp.GRB.Param.OutputFlag, flag)
    
        return model, x
        
            
        
    
    
    
    
   