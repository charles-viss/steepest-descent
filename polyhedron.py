import numpy as np
import sympy
import gurobipy as gp
import contextlib
import time

from polyhedral_model import PolyhedralModel
from utils import result, EPS, INF, METHODS


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
        self.model = None
        print('Problem size: n = {},  m_B = {},  m_A = {}'.format(self.n, self.m_B, self.m_A))
    
    # get active constraints at given solution
    def get_active_constraints(self, x):
        B_x = self.B.dot(x)
        inds = []
        for i in range(self.m_B):
            if self.d[i] - B_x[i] <= EPS:
                inds.append(i)
        #print('Active inds:')
        #print(inds)        
        #print('Differences:')
        #for i in inds:
        #    print(self.d[i] - B_x[i])
        return inds   

    
    # construct polyhedral model for computing circuits
    def build_polyhedral_model(self, x=None, primal=True, method='dual_simplex'):
        active_inds = []
        if x is not None:
            active_inds = self.get_active_constraints(x)
        pm = PolyhedralModel(B=self.B, A=self.A, c=self.c, active_inds=active_inds, method=method)
        return pm
    
	
	#given a point x in P with feasible direction g, compute the maximum step size alpha
    def get_max_step_size(self, x, g, active_inds=None, y_pos=None):
	
        inds = range(self.m_B)
        if y_pos is None:
            inds = [i for i in range(self.m_B) if y_pos[i] > 0.0]
        inds = [i for i in inds if i not in active_inds]

        alpha = float('inf')
        active_ind = None
        for i in inds:
            B_g_i = y_pos[i] if y_pos is not None else self.B[i].dot(g)
            if B_g_i <= EPS:
                continue
            B_x_i = self.B[i].dot(x)
            a = (self.d[i] - B_x_i) / float(B_g_i)
            if a <= alpha:
                alpha = a 
                active_ind = i
        return alpha, active_ind
    

    # build a gurobi LP for the polyhedron          
    def build_gurobi_model(self, c=None, verbose=False, method='primal_simplex'):
		
        if c is None:
            c = self.c
        assert c is not None, 'Provide an objective function'
		
        self.model = gp.Model()
        self.x = []
        for i in range(self.n):
            self.x.append(self.model.addVar(lb=-INF, ub=INF, name='x_{}'.format(i)))
        for i in range(self.m_A):
            self.model.addConstr(gp.LinExpr(self.A[i], self.x) == self.b[i], name='A_{}'.format(i))
        for i in range(self.m_B):
            self.model.addConstr(gp.LinExpr(self.B[i], self.x) <= self.d[i], name='B_{}'.format(i))
            
        self.set_objective(c)     
        self.set_verbose(verbose)		
        self.set_method(method)

    
    # (re)set objective function
    def set_objective(self, c):
        self.c = c
        if self.model is not None:
            self.model.setObjective(gp.LinExpr(self.c, self.x))

          
    # change model verbose settings
    def set_verbose(self, verbose):
        flag = 1 if verbose else 0
        with contextlib.redirect_stdout(None):
            self.model.setParam(gp.GRB.Param.OutputFlag, flag)
            
            
    def set_method(self, method):
        self.method = method
        with contextlib.redirect_stdout(None):
            self.model.Params.method = METHODS[method]
            
            
    # find a feasible solution within the polyhedron
    def find_feasible_solution(self, verbose=False):
        
        c_orig = np.copy(self.c)
        c = np.zeros(self.n)
        if self.model is None:
            self.build_gurobi_model(c, verbose)
        else:
            self.set_objective(c)           
        
        self.model.optimize()
        if self.model.status != gp.GRB.Status.OPTIMAL:
            raise RuntimeError('Failed to find feasible solution.')
        
        self.set_objective(c_orig)
        x_feasible = self.model.getAttr('x', self.x)        
        return x_feasible
        
        
    # sovle linear program using traditional simplex method; option to give warm started model
    def solve_lp(self, c=None, verbose=False, record_objs=True):
        
        if c is None:
            assert self.c is not None, 'Need objective function'
            c = self.c
         
        if self.model is None:
            self.build_gurobi_model(c=c)
        self.set_objective(c)
        t0 = time.time()
            
        obj_values = []
        def obj_callback(model, where):
            if where == gp.GRB.Callback.SIMPLEX:
                obj = model.cbGet(gp.GRB.Callback.SPX_OBJVAL)
                obj_values.append(obj)
                
        if record_objs:
            self.model.optimize(obj_callback)
        else:
            self.model.optimize()
        if self.model.status != gp.GRB.Status.OPTIMAL:
            raise RuntimeError('Model failed to solve')
            
        x_optimal = self.model.getAttr('x', self.x)        
        obj_optimal = self.model.objVal
        num_steps = self.model.getAttr('IterCount')
        #solve_time = self.model.getAttr('Runtime')
        solve_time = time.time() - t0
        output = result(0, x=x_optimal, obj=obj_optimal, n_iters=num_steps, solve_time=solve_time,
                        obj_values=obj_values)        
        return output
        
                
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
        #    raise ValueError('The direction {} is not a circuit of P'.format(g.T))
        circuit = np.array(ker_D[0]).reshape(self.n)
        
        #normalize
        circuit= circuit*sympy.lcm([circuit[i].q for i in range(self.n) if circuit[i] != 0])
        
        #make sure circuit has correct sign
        for i in range(self.n):
            if abs(g[i]) >= EPS:
                if circuit[i]*g[i] < 0:
                    circuit = -1*circuit
                break
        return circuit
    
    
   
