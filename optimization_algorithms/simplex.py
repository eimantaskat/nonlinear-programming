import numpy as np

def simplex(fw, x0, r, alpha, tol=1e-6):
    iter_count = 0

    # create initial simplex
    n = len(x0)
    simplex = np.zeros((n+1, n))
    simplex[0] = np.array(x0)

    delta1 = (np.sqrt(n + 1) + n - 1) / (n * np.sqrt(2)) * alpha
    delta2 = (np.sqrt(n + 1) - 1) / (n * np.sqrt(2)) * alpha
    for i in range(1, n+1):
        for j in range(n):
            if i == j + 1:
                simplex[i][j] = simplex[0][j] + delta2
            else:
                simplex[i][j] = simplex[0][j] + delta1

    # iterate until convergence
    while True:
        iter_count += 1
        # find worst and best points
        worst = 0
        best = 0
        for i in range(1, n+1):
            if fw.at(simplex[i], r) > fw.at(simplex[worst], r):
                worst = i
            if fw.at(simplex[i], r) < fw.at(simplex[best], r):
                best = i

        # calculate new point
        xc = np.zeros(n)
        for i in range(n+1):
            if i != worst:
                xc += simplex[i]
        xc /= n

        # reflect worst point
        xr = -simplex[worst] + 2 * xc

        # check if reflected point is better than worst point
        if fw.at(xr, r) < fw.at(simplex[worst], r):
            simplex[worst] = xr
        else:
            # contract simplex
            xc = (simplex[worst] + xc) / 2
            for i in range(n+1):
                if i != worst:
                    simplex[i] = (simplex[i] + simplex[worst]) / 2
                if fw.at(simplex[i], r) < fw.at(simplex[best], r):
                    best = i
            if fw.at(xr, r) < fw.at(simplex[best], r):
                simplex[worst] = xr
            else:
                # shrink simplex towards best point
                for i in range(n+1):
                    if i != best:
                        simplex[i] = (simplex[i] + simplex[best]) / 2
                    if fw.at(simplex[i], r) < fw.at(simplex[best], r):
                        best = i

        # check if simplex is small enough
        if np.linalg.norm(simplex[worst] - simplex[best]) < tol:
            break

    # return best point
    return simplex[best], iter_count
