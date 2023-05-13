import numpy as np
import sympy as sp

from optimization_algorithms import simplex
from function_wrapper import FunctionWrapper


if __name__ == '__main__':
	symbols = ['x', 'y', 'z', 'r']

	x, y, z, r = sp.symbols(symbols)
	gi = 2*x*y + 2*x*z + 2*y*z - 1

	penalty_func = -x*y*z + 1/r * (sp.Max(0, -x)**2 + sp.Max(0, -y)**2 + sp.Max(0, -z)**2 + sp.Pow(gi, 2))

	penalty_function = FunctionWrapper(function=penalty_func, symbols=['x', 'y', 'z', 'r'])

	x_values = np.array([[0]*3, [1]*3, [.2, .5, 0]])
	tol = 1e-4
	for x_val in x_values:
		x0 = x_val
		alpha = 0.1
		r_val = 10
		iter_count = 0
		while r_val > tol:
			x0, iter = simplex(penalty_function, x0, r_val, alpha=alpha, tol=tol)
			iter_count += iter
			r_val /= 10

		print(f'Starting point: {x_val}')
		print(f'Optimal point: {x0}')
		print(f'Number of iterations: {iter_count}')
		print(f'Number of function evaluations: {penalty_function.function.times_called}')
		print(f'Value of the penalty function at the point: {penalty_function.at(x0, r_val)}\n')
