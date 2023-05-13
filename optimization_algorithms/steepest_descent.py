import numpy as np

def golden_line_search(f, x, d, l=0, r=1, eps=1e-6):
    tau = (np.sqrt(5) - 1) / 2
    big_l = r - l
    x1 = r - big_l * tau
    x2 = l + big_l * tau
    while big_l > eps:
        if f(x + x2 * d) < f(x + x1 * d):
            l = x1
            big_l = r - l
            x1 = x2
            x2 = l + big_l * tau
        else:
            r = x2
            big_l = r - l
            x2 = x1
            x1 = r - big_l * tau
    return (l + r) / 2

def steepest_descent(fw, x0, tol=1e-6):
    iter_count = 0
    x = x0
    while True:
        iter_count += 1
        grad = fw.gradient_at(x)
        def f(gamma):
            return fw.at(x + gamma * -grad)
        step_size = golden_line_search(f, 0, 1)
        x_new = x - step_size * grad
        if np.linalg.norm(x - x_new) < tol:
            break
        x = x_new
    return x_new, iter_count
