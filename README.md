# steepest-descent

An implementation of the steepest-descent circuit augmentation scheme for solving general linear programs, as described in the paper: https://arxiv.org/abs/1911.08636. The basic principle is as follows: Given an initial feasible solution x_0 in a polyhedron P = {x: Ax = b, Bx <=d}, the steepest-descent scheme solves LP = min{c^T(x): x in P}. At each iteration, the algorithm computes a so-called _steepest-descent_ direction y_i at the current solution x_i. Taking a maximal step along this direction yields an improved solution x_{i+1} = x_i + alpha_i * y_i, and the scheme terminates once the steepest-descent direction y_i is no longer a strictly improving search direction. This scheme generalizes the minimum-mean cycle canceling algorithm to general linear programs and offers several promising convergence bounds.

The purpose of this repo is to provide an actual implementation of the steepest-descent scheme, one of several _circuit augmentation schemes_ for solving linear programs. As a generalization of the edge directions of polyhedra, the circuits of polyhedra play a fundamental role in the theory of linear programming. Augmentation schemes which take steps along circuits generalize the simplex method; however, unlike the simplex method, the schemes are not restricted to only the edges of a polyhedron and may traverse its interior. Thus, the _Polynomial Hirsch Conjecture_ need not be true for there to exist a strongly-polynomial time circuit augmentation scheme. Additionally, circuits have natural combinatorial interpretations for many classical families of linear programs.

The challenge in implementing most circuit augmentations schemes lies in computing the required circuits. However, as shown in https://arxiv.org/abs/1811.00444, a polyhedral model for the set of circuits facilitates the efficient computation of a steepest-descent circuit via a linear programming oracle. We use a dynamic version of this oracle to compute these steepest-descent circuits, with each computed direction serving as a warm-start for the program in the following iteration. This repo therefore provides an opportunity to explore the behavior of the steepest-descent augmentation scheme when applied to real-world problems. We compare its performance to that of the simplex method, and compare our proposed approach to other possible implementations.

## Requirements

Requires Python 3.5+ and the Gurobi Optimizer (https://www.gurobi.com/), along with the following modules:
* GurobiPy
* numpy
* cvxopt

## Usage

The steepest-descent scheme can be launched via _main.py_, which reads in a linear program in MPS format and solves it via the steepest-descent scheme as well as the simplex method. See the well-known problems of the Netlib LP Test Set (https://www.netlib.org/lp/data/index.html), a library which serves as a benchmark for evaluating linear programming algorithms, for example problems in MPS format. We include a subset of these Netlib problems in the folder _netlib_lp_subset_. The program can also be used to test the steepest-descent algorithm on randomly generated _partition_polytopes_ or _spindles_ if an MPS file is not provided.

The possible arguments for _main.py_ are given below:

```
usage: main.py [-h] --mps_fn MPS_FILE --sd_method SD_METHOD --results_dir RESULTS_DIR --max_time MAx_TIME [--reset] \
                    [--partition_polytope] --n N --k K \
                    [--spindle] --spindle_dim SPINDLE_DIM --n_cone_facets N_CONE_FACETS --n_parallel_facets N_PARALLEL_FACETS

arguments:
  -h, --help            show this help message and exit
  --mps_fn MPS_FN
                        Path to MPS file
  --sd_method SD_METHOD
                        LP method used to compute steepest-descent direction at each iteration. (default is dual_simplex)
                        Options: dual_simnplex, primal_simplex, barrier.
  --results_dir RESULTS_DIR
                        Path to directory where computational results will be saved. (default is _results_)
  --max_time MAX_TIME
                        Maximum time (in seconds) for the steepest-descent scheme to run before it is terminated.
                        (default is 300)
                        
  --partition_polytope
                        If an MPS file is not given, use this flag to run the steepest-descent algorithm on a randomly generated
                        partition polytope.
  --n N
                        Number of items to be considered for the generated partition polytope.
  --k K
                        Number of clusters to be considered for the generated partition polytope.

  --spindle
                        If an MPS file is not given, use this flag to run the steepest-descent algorithm on a randomly generated
                        spindle (the intersection of two opposing cones).
  --spindle_dim SPINDLE_DIM
                        Dimension of the generated spindle.
  --n_cone_facets N_CONE_FACETS
                        Number of facets which form each cone of the spindle.
  --n_parallel_facets N_PARALLEL_FACETS
                        Number of pairs of additional parallel facets to be added to spindle.
```

An test of the program can be run with the following command:
```
    python main.py --mps_fn netlib_lp_subset/adlittle --sd_method dual_simplex --results_dir results
```

See also the notebooks _run_tests.ipynb_ and _view_results.ipynb_ for examples of running the algorithm on multiple problems and visualizing the results.


