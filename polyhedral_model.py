import numpy as np
import gurobipy as gp

INF = 10e100
METHODS = {'primal_simplex': 0,
          'dual_simplex': 1,
          'barrier': 2,
          'concurrent': 3,
          'deterministic_concurrent': 4,
          'deterministic_concurrent_simplex': 5,}

class PolyhedralModel():
    
    # Given matrices B and A (and optional inds argument / objective function),
    # builds a polyhedral model for computing steepest-descent circuits
    # as a gurobi linear program model
    def __init__(self, B, A=None, active_inds=[], c=None, primal=True, method='primal_simplex'):
        
        self.model = gp.Model()
        self.primal = primal
        
        # add variables and constraints to the model
        self.m_B, self.n = B.shape
        if self.primal:
            self.x = []
            self.y_pos = []
            self.y_neg = []
            for i in range(self.n):
                self.x.append(self.model.addVar(lb=-INF, ub=INF, name='x_{}'.format(i)))
            for i in range(self.m_B):
                self.y_pos.append(self.model.addVar(lb=0.0, ub=1.0, name='y_pos_{}'.format(i)))
                self.y_neg.append(self.model.addVar(lb=0.0, ub=1.0, name='y_neg_{}'.format(i)))
                self.model.addConstr(gp.LinExpr(list(B[i]) + [-1, 1], 
                                                self.x + [self.y_pos[i], self.y_neg[i]]) == 0, 
                                     name='B_{}'.format(i))
    
            self.model.addConstr(gp.LinExpr([1]*(2*self.m_B), self.y_pos + self.y_neg) == 1, 
                                 name='1_norm')
                
            if A is not None:
                self.m_A = A.shape[0]
                for i in range(self.m_A):
                    self.model.addConstr(gp.LinExpr(A[i], self.x) == 0, 
                                         name='A_{}'.format(i))
            else:
                self.m_A = 0
                
            if c is not None:
                self.set_objective(c)
            self.set_active_inds(active_inds)
            self.set_method(method)
                
        else:
            raise RuntimeError('Not yet implemented')
            
        self.model.update()
                
                
    def set_objective(self, c):
        self.c = c
        self.model.setObjective(gp.LinExpr(c, self.x))
            
    def set_active_inds(self, active_inds):
        self.active_inds = active_inds
        for i in range(self.m_B):
            self.y_pos[i].ub = 1.0
        for i in self.active_inds:
            self.y_pos[i].ub = 0.0
                
    def set_method(self, method):
        self.method = method
        self.model.Params.method = METHODS[method]
          
    # warm start the model with the provided solution   
    def set_solution(self, g):
        for i in range(self.n):
            self.x[i].lb = g[i]
            self.x[i].ub = g[i]
        for i in range(self.m_B):
            self.y_pos[i].ub = 1.0
            
        # solve the modified model to obtain desired solution    
        self.model.Params.method = METHODS['deterministic_concurrent_simplex']
        self.model.optimize()
        
        # reset the model contraints back to its original state
        for i in range(self.n):
            self.x[i].lb = -INF
            self.x[i].ub = INF
        self.set_active_inds(self.active_inds)
        self.set_method(self.method)
        self.model.update()
        
                
    def compute_sd_direction(self, verbose=False):
        flag = 1 if verbose else 0
        self.model.setParam(gp.GRB.Param.OutputFlag, flag)
        
        self.model.optimize()
        if self.model.status != gp.GRB.Status.OPTIMAL:
            raise RuntimeError('Model failed to solve')
            
        g = self.model.getAttr('x', self.x)
        steepness = self.model.objVal
        num_steps = self.model.getAttr('IterCount')
        solve_time = self.model.getAttr('Runtime')
        
        return np.asarray(g), steepness, num_steps, solve_time
                
    def reset(self):
        self.model.reset()


            
#B = np.array([
#        [1, 0, 0],
#        [0, 0, -1],
#        [0, -1, -1],
#        [0, -1, 0],
#        [2, 1, 1],
#        [3, 3, -1]]) 
#A = np.array([[0, 1, -1]])
#    
#c = np.array([-8, -1, -5])
#x_initial = np.array([4./18, -3./18, 3./18])
#active_inds = [0]
#
#pm = PolyhedralModel(B=B, c=c, active_inds=active_inds, A=A)
##pm.set_solution(x_initial)
#g, s = pm.compute_sd_direction()
#        
#print('Steepest-descent circuit:')
#print(g)
#
#print('Steepness')
#print(s)