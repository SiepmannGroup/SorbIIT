from email.mime import base
import os
import numpy as np


'''
Coulomb constant in atomic units (angstom, K)
'''
k_col=(1.602176565E-19**2)*1e10/(4*np.pi)/8.854187817E-12/1.3806488E-23


'''
Attributes of trajectory files
'''
attrs = ['cell_angle', 'cell_length', 'frames', 'grid_sizes', 'ortho_length', 'translation_vector']


class Atom:
    '''
    Atom with LJ 12-6 and coulomb interactions.
    Used for combining ZTB grids into energies
    '''
    def __init__(self, name, sigma, epsilon, q, atomid=-1):
        self.name = name
        self.sigma = sigma
        self.epsilon = epsilon
        self.q = q
        self.atomid = atomid

    def __mul__(self, other):
        sigma_ij = (self.sigma + other.sigma) / 2
        epsilon_ij = (self.epsilon * other.epsilon) ** 0.5
        qiqj = self.q * other.q
        return np.array([4 * epsilon_ij * sigma_ij ** 12, -4 * epsilon_ij * sigma_ij ** 6, qiqj * k_col])


class TraPPE:
    '''
    Trappe force field with predefined parameters
    '''
    # TraPPE-zeo
    Si_z = Atom('Si', 2.3, 22, 1.5, 177)
    Al_z = Atom('Al', 2.3, 22, 1.5, 177)
    P_z = Atom('P', 2.3, 22, 1.5, 177)
    O_z = Atom('O', 3.3, 53, -0.75, 178)
    # TraPPE-UA
    CH4 = Atom("CH4", 3.73, 148, 0, 3)
    CH3 = Atom("CH3", 3.75, 98, 0, 4)
    CH2 = Atom("CH2", 3.95, 46, 0, 5)
    CH = Atom("CH", 4.68, 10, 0, 6)
    # TraPPE-Small
    O_CO2 = Atom("O", 3.05, 79, -0.35, 51)
    C_CO2 = Atom("C", 2.8, 27, 0.7, 52)
    S_H2S = Atom("S", 3.6, 122, 0, 11)
    H_H2S = Atom("H", 2.5, 50, 0.21, 12)
    X_H2S = Atom("X", 0, 0, -0.42, 13)
    # Rare gases
    Kr = Atom("Kr", 3.63, 166.4, 0, 501)  #Kr from Talu2001
    Xe = Atom("Xe", 4.10, 221, 0, 502) #Xe from Hirshfelder,Burtiss,Bird entry with ref J


Z_ATOMS = {'Si': TraPPE.Si_z, 'O': TraPPE.O_z, 'Al': TraPPE.Al_z, 'P': TraPPE.P_z}

def read_ztb(filename, verbose=0):
    '''
    Reads a ZTB tabulated potential file specified by FILENAME.
    Returns a dictionary ZTB_PROPS describing the properties of the
    ZTB file, and a Pandas Dataframe ZTB_Table containing the tabulated
    potential data.
    Prints relavant information if VERBOSE=1.
    '''
    ztb_props_name = ['cell_parameters',
             'angles',
             'grid_count',
             'atom_type',
             'use_ewald',
             'use_tail_correction',
             'use_shifted_potential',
             'r_cutoff']
    ztb_props_type = [np.float, np.float, np.int32, np.int32, np.int32, np.int32, np.int32, np.float]
    ztb_props_count = [3, 3, 3, 1, 1, 1, 1, 1]
    ztb_props_size = [8, 8, 4, 4, 4, 4, 4, 8]
    
    lj_factor = 20000
    props_data_temp = []
    ztb_props = {}
    data_raw = np.fromfile(filename, np.uint8)
    bytes_read = 0
    for i in range(len(ztb_props_name)):
        props_data_temp.append(
            np.frombuffer(data_raw, ztb_props_type[i],
                count=ztb_props_count[i], offset=bytes_read)
        )
        bytes_read += ztb_props_size[i] * ztb_props_count[i]
    str_size = 128
    atoms = []
    for i in range(props_data_temp[3][0]):
        atom_name = [chr(x) for x in np.frombuffer(
            data_raw, np.uint8, count=str_size, offset=bytes_read)]
        atoms.append(''.join(atom_name).strip())
        bytes_read += str_size
    if verbose:
        print('Successfully loaded ZTB data properties:')
    for i in range(len(ztb_props_name)):
        prop_item = props_data_temp[i]
        if ztb_props_count[i] == 1:
            prop_item = prop_item[0]
        else:
            prop_item = list(prop_item)
        if i == 3:
            prop_item = atoms
        if i >= 4 and i < 7:
            assert abs(prop_item) == 0 or abs(prop_item) == 1, 'Invalid Boolean value!'
            prop_item = bool(prop_item)
        if verbose:
            print('\t'+ztb_props_name[i] + ': ' + str(prop_item))
        ztb_props[ztb_props_name[i]] = prop_item

    ngrid = ztb_props['grid_count']
    npoints = ngrid[0] * ngrid[1] * ngrid[2]
    ndata = npoints * len(atoms) * 3
    table_data = np.frombuffer(data_raw, np.float, offset=bytes_read)
    assert len(table_data) == ndata,\
            'Error: ' + str(ndata) + 'items should be loaded, found ' + str(len(table_data)) + ' items,'
    table_data = np.array(table_data).reshape((npoints, 3 * len(atoms)))
    table_data[:, 0::3] /= lj_factor ** 2
    table_data[:, 1::3] /= lj_factor
    grids = {}
    for i in range(len(atoms)):
        grids[atoms[i]] = table_data[:, i*3 : (i+1)*3].reshape(ztb_props['grid_count'] + [3], order='F')
    if verbose:
        print('Successfully loaded ' + str(npoints) + ' grid points for ' + str(len(atoms)) + ' atoms from ZTB file.')
    return ztb_props, grids


def calc_energies(grid, f_atoms, g_atoms, emin, emax):
    egrids = {}
    for g in g_atoms:
        egrids[g.name] = None
        for f in f_atoms:
            w = (f * g).reshape((1, 1, 1, 3))
            if egrids[g.name] is None:
                egrids[g.name] = np.sum(w * grid[f.name], axis=-1)
            else:
                egrids[g.name] += np.sum(w * grid[f.name], axis=-1)
        egrids[g.name] = np.clip(egrids[g.name], emin, emax)
    return egrids

def get_ztb(path, key, g_atoms, emin=-10000, emax=10000, calc_energy=True):
    metadata, grid = read_ztb(os.path.join(path, "%s.ztb" % key))
    ncell = 1
    with open(os.path.join(path, "../cif/%s.cif" % key), 'r') as f:
        line = f.readline().strip("#").split()
        ncell *= int(line[0]) * int(line[1]) * int(line[2])
    if calc_energy:
        egrid = calc_energies(grid, 
                [Z_ATOMS[x] for x in metadata['atom_type']], g_atoms, emin=emin, emax=emax)
        return metadata, egrid, ncell
    else:
        return metadata, grid, ncell
    

def get_traj(h5file, key):
    shape = h5file[key].get('grid_sizes') or h5file[key].attrs.get('grid_sizes')
    shape = np.array(shape)
    traj = {}
    frames = h5file[key].get("frames") or h5file[key].attrs.get("frames")
    metadata = {}
    for u, arr in h5file[key].items():
        if u in attrs:
            metadata[u] = arr
        else:
            indices = arr[0]
            values = arr[1] / frames
            densearr = np.zeros(shape[0] * shape[1] * shape[2])
            densearr[indices.astype(np.int)] = values
            densearr = densearr.reshape(shape)
            traj[u] = densearr
    for k, v in h5file[base].attrs.items():
        metadata[k] = np.array(v)
    return metadata, traj

def get_traj_all(h5file, key, ps, ts):
    base_key = key + "-P0-T0"
    shape = h5file[base_key].get('grid_sizes') or h5file[base_key].attrs.get('grid_sizes')
    shape = np.array(shape)
    metadata = {}
    for k, v in h5file[base_key].items():
        if k in attrs:
            metadata[k] = np.array(v)
    for k, v in h5file[base_key].attrs.items():
        metadata[k] = np.array(v)
    gridsize = shape[0] * shape[1] * shape[2]
    indices = []
    values = []
    for p in range(ps):
        for t in range(ts):
            offset = (p * ts + t) * gridsize
            local_key = "%s-P%d-T%d" % (key, p, t)
            frames = h5file[local_key].get("frames") or h5file[local_key].attrs.get("frames")
            indices = np.concatenate([indices, offset + h5file[local_key]['COM'][0]])
            # still need to be divided by the number of unit cells
            values = np.concatenate([values, h5file[local_key]['COM'][1] / frames])
    traj = np.zeros(ps * ts * gridsize)
    traj[indices.astype(np.int)] = values
    traj = traj.reshape([ps * ts] + shape.tolist())
    return metadata, traj
                    
    
        
