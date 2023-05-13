import numpy as np
import sympy as sp

from optimization_algorithms import simplex
from function_wrapper import FunctionWrapper


if __name__ == '__main__':
	symbols = ['x', 'y', 'z', 'r']

	x, y, z, r = sp.symbols(symbols)
	gi = 2*x*y + 2*x*z + 2*y*z - 1

	penalty_func = -x*y*z + 1/r * (sp.Max(0, -x)**2 + sp.Max(0, -y)**2 + sp.Max(0, -z)**2 + sp.Pow(gi, 2))

	# foo = sp.sympify(penalty_func)
	# result = foo.subs([(x, 6**0.5/6), (y, 6**0.5/6), (z, 6**0.5/6), (r, 1)])

	penaltyFunction = FunctionWrapper(function=penalty_func, symbols=['x', 'y', 'z', 'r'])
	# print(penaltyFunction.at([1]*3, 0.2)) # sqrt(6)/6
	# print(penaltyFunction.at([6**0.5/6]*3 + [0.2])) # sqrt(6)/6

	x_val = np.array([1, 1, 1])
	r_val = 0.1
	result, iter_count = simplex(penaltyFunction, x_val, r_val, alpha=.3, tol=1e-4)
	print(result, iter_count)