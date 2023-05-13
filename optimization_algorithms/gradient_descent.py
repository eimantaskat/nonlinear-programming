import numpy as np

def gradient_descent(fw, x0, step_size, tol):
    iter_count = 0
    x = x0
    while True:
        iter_count += 1
        x_new = x - step_size * fw.gradient_at(x)
        if np.linalg.norm(x - x_new) < tol:
            break
        x = x_new
    return x_new, iter_count
