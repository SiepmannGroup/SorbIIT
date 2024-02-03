import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import curve_fit
import torch

def langmuir(x_fit, logq, h, s):
    logp, invT = x_fit[0], x_fit[1]
    logit = s + h * invT + logp
    return np.exp(logq) / (1 + np.exp(-logit))

def quadratic(x_fit, logq, h1, s1, h2, s2):
    logp, invT = x_fit[0], x_fit[1]
    k1 = np.exp(h1 * invT + s1 + logp)
    k2 = np.exp(h2 * invT + s2 + 2 * logp)
    theta = (k1 + 2 * k2) / (1 + k1 + k2)
    return np.exp(logq) * theta

def total_adsorption(voxels_all, pool):
    # use unit of molec / AA^3
    # 1 molecule per cubic angstrom = 1661 mol / L
    vol_cell = (0.1 * pool) ** 3
    y_tot = torch.mean(voxels_all.view(voxels_all.shape[1], -1), 1) / vol_cell
    return y_tot

def langmuir_coefficients(popt, logp, invT):
    # differentiating quadratic isotherm
    Q = np.exp(popt[0])
    h, s = tuple(popt[1:])
    k = np.exp(h * invT + s + logp)
    theta = k / (1 + k)
    # Coefficients: dQ, dS, dH
    c0 = theta
    c1 = theta * (1 - theta)
    c2 = c1 * invT
    return Q, np.stack([c0, c1, c2], axis=1)

def quadratic_coefficients(popt, logp, invT):
    # differentiating quadratic isotherm
    Q = np.exp(popt[0])
    h1, s1, h2, s2 = tuple(popt[1:])
    k1 = np.exp(h1 * invT + s1 + logp)
    k2 = np.exp(h2 * invT + s2 + 2 * logp)
    theta = k1 / (1 + k1 + k2)
    gamma = 2 * k2 / (1 + k1 + k2)
    # Coefficients: dQ, dS1, dH1, dS2, dH2
    c0 = theta + gamma
    c1 = theta * (1 - theta - gamma)
    c2 = c1 * invT
    c3 = gamma * (1 - (theta + gamma) / 2)
    c4 = c3 * invT
    return Q, np.stack([c0, c1, c2, c3, c4], axis=1)

def make_solver(f_total, f_coeffs):
    def solve(log_p, inv_t, voxels_all, pool, return_total=False):
        n_states = len(log_p) * len(inv_t)
        p_fit, t_fit = np.meshgrid(log_p, inv_t, indexing='ij')
        x_fit = np.stack([p_fit, t_fit]).reshape(2, -1)
        vol_cell = (0.1 * pool) ** 3
        y_tot = total_adsorption(voxels_all, pool).numpy()
        popt, _ = curve_fit(f_total, x_fit, y_tot, bounds=(-5000, 5000), loss='linear', max_nfev=50000)

        grid_sizes = voxels_all.shape[2:]
        # only regress non-zero entries
        mask = torch.any(voxels_all.reshape(-1, *grid_sizes) > 0, 0).flatten()
        void_frac = mask.float().mean()
        # Coefficients: dQ, dS1, dH1, dS2, dH2
        Q, A = f_coeffs(popt, x_fit[0], x_fit[1])
        B = voxels_all.numpy().reshape(n_states, -1) / (Q * vol_cell) # need to convert density-based Q back to number
        X = torch.zeros((A.shape[1], B.shape[1]))
        B = B[:, mask]
        X_active, _, _, _ = lstsq(A, B)
        X[:, mask] = torch.from_numpy(X_active.astype(np.float32))
        if return_total:
            y_tot_pred = f_total(x_fit, *popt)
            return X.view(-1, *grid_sizes), popt, y_tot, y_tot_pred
        else:
            return X.view(-1, *grid_sizes), popt, void_frac
    return solve

def make_predictor(f_coeffs):
    def predict(log_p, inv_t, X, popt, pool):
        p_fit, t_fit = np.meshgrid(log_p, inv_t, indexing='ij')
        x_fit = np.stack([p_fit, t_fit]).reshape(2, -1)
        grid_sizes = X.shape[1:]
        vol_cell = (0.1 * pool) ** 3
        Q, A = f_coeffs(popt, x_fit[0], x_fit[1])
        A = torch.tensor(A).float()
        voxels_all = (A @ X.view(A.shape[1], -1)).view(A.shape[0], *grid_sizes) * Q * vol_cell
        return voxels_all
    return predict

solve_langmuir = make_solver(langmuir, langmuir_coefficients)
solve_quadratic = make_solver(quadratic, quadratic_coefficients)
predict_langmuir = make_predictor(langmuir_coefficients)
predict_quadratic = make_predictor(quadratic_coefficients)