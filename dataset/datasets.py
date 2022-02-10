import h5py
import numpy as np
import torch
from scipy.optimize import curve_fit
import scipy
from torch.utils.data import Dataset
from utils import CellGrid3, get_ztb, get_traj_all, quadratic, langmuir


# profile flag
PROFILE = False
 
log = lambda x: PROFILE and print(x)

class SpatialDerivativeDataset(Dataset):

    # global data normalization
    x_mean = x_std = 6400
    y_ext_mean = -20.0
    y_ext_std = 8.0
    y_int_std = 30.0

    EPS = 1e-10


    def __init__(
        self,
        g_atoms,
        pool,
        sizes,
        movie_path, ztb_path,
        pressures,
        temperatures,
        keys=None,
        ntrans=32,
        nrot=0,
        axisrot=True,
        shuffle=False,
        cif_path=None,
        positional_encoding=False,
        normalize_all=False,
        symmetrize_in_place=True,
        return_symmetrize_func=False,
    ):
        self.atoms = g_atoms
        self.pool = pool
        self.movie_file = movie_path
        self.ztb_path = ztb_path        
        self.ntrans = ntrans
        self.nrot = nrot
        self.axisrot = axisrot
        self.shuffle = shuffle
        if keys is None:
            with h5py.File(self.movie_file, 'r') as h5file:
                self.keys = np.unique([x[:-6] for x in h5file.keys()])
        else:
            self.keys = keys
        self.pressures = pressures
        self.temperatures = temperatures
        self.log_p = np.log(np.array(pressures))
        self.inv_t = 1000 / np.array(temperatures)
        if shuffle:
            np.random.shuffle(self.keys)
        self.cif_path = cif_path
        self.cache = {}
        self.in_size, self.out_size = sizes
        if self.in_size < self.out_size:
            raise ValueError("Output size should not be larger than input!")
        self.positional_encoding = positional_encoding
        self.normalize_all = normalize_all
        self.return_symmetrize_func = return_symmetrize_func
        self.symmetrize_in_place = symmetrize_in_place

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        try:
            if self.return_symmetrize_func:
                return self.get_unit_cells(index)
            else:
                return self.sample(index, self.ntrans, self.nrot, self.axisrot)
        except KeyError:
            return None
    
    def process(self, x, y):
        if x.dim() == 4:
            x[:2] = -(x[:2] - self.x_mean) / self.x_std
            if self.normalize_all:
                y[0] = (torch.log(torch.clamp(y[0], min=self.EPS)) - self.y_ext_mean) / self.y_ext_std
                y[1:] = y[1:] / self.y_int_std
        else:
            x[:, :2] = -(x[:, :2] - self.x_mean) / self.x_std
            if self.normalize_all:
                y[:, 0] = (torch.log(torch.clamp(y[:, 0], min=self.EPS)) - self.y_ext_mean) / self.y_ext_std
                y[:, 1:] = y[:, 1:] / self.y_int_std
        return x, y
            
    def set_keys(self, keys):
        self.keys = keys
        
    def subset(self, keys):
        subset = SpatialDerivativeDataset(
            self.atoms, self.pool, (self.in_size, self.out_size),
            self.movie_file, self.ztb_path,
            self.pressures, self.temperatures, 
            self.ntrans, self.nrot, self.axisrot, self.shuffle, self.cif_path)
        subset.set_keys(keys)
        return subset
            
    def solve_derivatives(self, key, ncell):
        log("--SOLVE: reading voxels---")
        ps, ts = len(self.log_p), len(self.inv_t)
        vol_cell = (0.1 * self.pool) ** 3
        with h5py.File(self.movie_file, 'r') as h5file:
            info, voxels_all = get_traj_all(h5file, key, ps, ts)
        grid_sizes = np.array(info["grid_sizes"]) // self.pool
        log("--SOLVE: pooling---")
        
        with torch.no_grad():
            voxels_all = torch.unsqueeze(torch.from_numpy(voxels_all), 0) / ncell
            voxels_all = torch.nn.functional.avg_pool3d(voxels_all, self.pool, self.pool, 0) * self.pool ** 3

        log("--SOLVE: global fit---")
        
        p_fit, t_fit = np.meshgrid(self.log_p, self.inv_t, indexing='ij')
        x_fit = np.stack([p_fit, t_fit]).reshape(2, -1)
        # use unit of molec / AA^3
        # 1 molecule per cubic angstrom = 1661 mol / L
        y_tot = torch.mean(voxels_all.view(ps * ts, -1), 1).numpy() / vol_cell 
        func = quadratic
        popt, pcov = curve_fit(func, x_fit, y_tot, bounds=(-5000, 5000), loss='linear', max_nfev=50000)
        y_pred = func(x_fit, *popt)
        #print("Zeolite:", key, "Capacity (mol/L):", 2 * np.exp(popt[0]) * 1661, "R2:", r2_score(y_tot, y_pred))
        log("--SOLVE: local fit---")
        # differentiating quadratic isotherm
        Q = np.exp(popt[0])
        h1, s1, h2, s2 = tuple(popt[1:])
        info["isotherm_coeffs"] = popt

        logp, invT = x_fit[0], x_fit[1]
        k1 = np.exp(h1 * invT + s1 + logp)
        k2 = np.exp(h2 * invT + s2 + 2 * logp)

        theta = k1 / (1 + k1 + k2)
        gamma = 2 * k2 / (1 + k1 + k2)
        
        ngrid = np.prod(grid_sizes)

        # only regress non-zero entries
        mask = torch.any(voxels_all.reshape([-1] + grid_sizes.tolist()) > 0, 0).flatten()

        # Coefficients: dQ, dS1, dH1, dS2, dH2
        c0 = theta + gamma
        c1 = theta * (1 - theta - gamma)
        c2 = c1 * invT
        c3 = gamma * (1 - (theta + gamma) / 2)
        c4 = c3 * invT
        A = np.stack([c0, c1, c2, c3, c4], axis=1)
        B = voxels_all.numpy().reshape(ps * ts, -1) / (Q * vol_cell) # need to convert density-based Q back to number
        X = torch.zeros((A.shape[1], B.shape[1]))
        B = B[:, mask]
        X_active, _, _, _ = scipy.linalg.lstsq(A, B)
        X[:, mask] = torch.from_numpy(X_active.astype(np.float32))
        
        return info, X.view(-1, *grid_sizes)
    
    
    def get_samplers(self, index):
        key = self.keys[index]
        log("--start get_ztb---")
        grid_info, egrid, ncell = get_ztb(self.ztb_path, key, self.atoms)
        log("--end get_ztb---")
        log("--start solve---")
        traj_info, tgrid_torch = self.solve_derivatives(key, ncell)
        log("--end solve---")
        log("--start building grid---")
        spacing = np.array(traj_info['ortho_length']) / np.array(traj_info['grid_sizes'])
        tgrid_trans = np.round(traj_info['translation_vector'] / spacing).astype(np.int).T
        
        tgrid_sampler = CellGrid3(tgrid_torch, tgrid_trans // self.pool)
        if self.cif_path is not None and self.symmetrize_in_place:
            pos_enc = tgrid_sampler.symmetrize(self.cif_path + "/%s.cif" % key,
                self.positional_encoding)
        
        egrid_torch = torch.cat([torch.unsqueeze(torch.from_numpy(egrid[g.name]), 0).float() for g in self.atoms], 0)
        egrid_torch = torch.nn.functional.avg_pool3d(torch.unsqueeze(egrid_torch, 0), self.pool, self.pool, 0)[0]
        egrid_sampler = CellGrid3(egrid_torch.float(), tgrid_trans // self.pool)
        if self.positional_encoding:
            egrid_sampler = CellGrid3(torch.cat([egrid_sampler.voxels, pos_enc], 0),
                tgrid_trans // self.pool, pad=False)
        log("--end building grid---")
        return egrid_sampler, tgrid_sampler

    def get_unit_cells(self, index):
        key = self.keys[index]
        egrid_sampler, tgrid_sampler = self.get_samplers(index)
        x, y = egrid_sampler.cart_cell, tgrid_sampler.cart_cell
        transform = egrid_sampler.get_symmetry_transforms(self.cif_path + "/%s.cif" % key, cartesian=None)
        symm_info = (
            torch.from_numpy(egrid_sampler.unitcell_grid(circular=False)).float(),
            torch.from_numpy(egrid_sampler.cell_vectors).float().contiguous(),
            torch.from_numpy(transform).float().contiguous(),
        )
        return (*self.process(x.unsqueeze(0).contiguous(), y.unsqueeze(0).contiguous()), symm_info)


    def sample(self, index, ntrans, nrot, axisrot):
        log("--start packing batch---")
        egrid_sampler, tgrid_sampler = self.get_samplers(index)
        
        translations = egrid_sampler.sample_translate(
            ntrans + nrot)
        if nrot:
            rotations = egrid_sampler.sample_rotate(nrot)
        dtrans = (self.in_size - self.out_size) // 2
        samples_in = []
        translate_in = egrid_sampler.get_translate(self.in_size, translations[:ntrans])
        samples_in.append(translate_in)
        if nrot:
            samples_in.append(egrid_sampler.get_rotate(self.in_size, rotations, translations[ntrans:]))
        if axisrot:
            samples_in.append(translate_in.permute((0, 1, 4, 2, 3)))
            samples_in.append(translate_in.permute((0, 1, 3, 4, 2)))
            
        samples_out = []
        translate_out = tgrid_sampler.get_translate(self.out_size, translations[:ntrans] + dtrans)
        samples_out.append(translate_out)
        if nrot:
            samples_out.append(tgrid_sampler.get_rotate(self.out_size, rotations, translations[ntrans:] + dtrans))
        if axisrot:
            samples_out.append(translate_out.permute((0, 1, 4, 2, 3)))
            samples_out.append(translate_out.permute((0, 1, 3, 4, 2)))
        
        samples_in = torch.cat(samples_in, 0).float()
        samples_out = torch.cat(samples_out, 0).float()
 
        log("--end packing batch---")
        return self.process(samples_in, samples_out)

        



        

