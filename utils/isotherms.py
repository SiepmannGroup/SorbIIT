import numpy as np


def langmuir(x_fit, logq, a, b):
    logp, invT = x_fit[0], x_fit[1]
    logit = a + b * invT + logp
    return np.exp(logq) / (1 + np.exp(-logit))

def quadratic(x_fit, logq, h1, s1, h2, s2):
    logp, invT = x_fit[0], x_fit[1]
    k1 = np.exp(h1 * invT + s1 + logp)
    k2 = np.exp(h2 * invT + s2 + 2 * logp)
    theta = (k1 + 2 * k2) / (1 + k1 + k2)
    return np.exp(logq) * theta

