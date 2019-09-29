import numpy
from scipy import optimize

from steepest_descent import steepest_descent_augmentation_scheme as sdac
from polyhedron import Polyhedron

A = None
b = None

B = numpy.array([
        [-1, 0, 0],
        [0, 0, -1],
        [0, -1, -1],
        [0, -1, 0],
        [2, 1, 1],
        [3, 3, -1]])

d = numpy.array([
        0,
        0,
        2,
        4,
        6,
        8])

m_B, n = B.shape
m_A = 0
#m_A = A.shape[0]

c = numpy.array([-8, -1, -5])

P = Polyhedron(B=B, d=d, A=A, b=b, c=c)
x_initial = numpy.array([0, 0, 0])
    
result = sdac(P, c, x_initial)
x_optimal = result.x

print('\nsolution using steepest-descent augmentation:')
print(result)
for i in range(len(result.circuits)):
    print('g_' + str(i) + ': ' + str(result.circuits[i].T) + ' with alpha_'
              + str(i) + ' = ' + str(result.steps[i]))

print('\noptimal solution using direct linear programming:')
#result = optimize.linprog(c=c, A_eq=A, b_eq=b, A_ub=B, b_ub=d, bounds=(None,None), method='simplex')
#print(result.x)
x_optimal, obj_optimal, num_steps, solve_time = P.solve_lp()
print('Optimal solution:')
print(x_optimal)
print('Optimal objective:')
print(obj_optimal)
print('Num steps:')
print(num_steps)
print('Solve time:')
print(solve_time)





