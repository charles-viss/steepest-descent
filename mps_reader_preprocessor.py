#source: https://github.com/sschnug/scipy_lp_dev
import numpy as np
from cvxopt.base import matrix, spmatrix
from cvxopt.modeling import op


def read_mps_preprocess(filepath):
    problem = op()
    problem.fromfile(filepath)
    mat_form = problem._inmatrixform(format='dense')
    format = 'dense'
    assert mat_form
    lp, vmap, mmap = mat_form

    variables = lp.variables()
    x = variables[0]
    c = lp.objective._linear._coeff[x]
    inequalities = lp._inequalities
    G = inequalities[0]._f._linear._coeff[x]
    h = -inequalities[0]._f._constant
    equalities = lp._equalities
    A, b = None, None
    if equalities:
        A = equalities[0]._f._linear._coeff[x]
        b = -equalities[0]._f._constant
    elif format == 'dense':
        A = matrix(0.0, (0,len(x)))
        b = matrix(0.0, (0,1))
    else:
        A = spmatrix(0.0, [], [],  (0,len(x)))  # CRITICAL
        b = matrix(0.0, (0,1))

    c = np.array(c).flatten()
    G = np.array(G)
    h = np.array(h).flatten()
    A = np.array(A)
    b = np.array(b).flatten()

    return c, G, h, A, b