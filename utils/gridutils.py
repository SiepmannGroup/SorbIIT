import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator
import torch
from torch.nn import functional as F


class CellGrid:

    def __init__(self, voxels, cell_vectors=None, pad=True):
        '''
        Initializes voxel grid of unit cell.
        If CELL_VECTORS is none, it is assumed that the grid
        is repeated by its edges along each dimension.
        If CELL_VECTORS is not none, it should be provided 
        in a UPPER triangular format where each COLUMN is 
        a cell vector.
        Update 2021-04-10:
        Make each grid hold a vector field instead of scalar (allow for multiple channels).
        The channel is located at the FIRST dimension (pytorch convention).
        Use entirely pytorch instead of numpy.
        '''
        self.voxels = voxels
        self.ndim = len(voxels.shape) - 1
        if cell_vectors is None:
            self.cell_vectors = np.diag(self.voxels.shape[1:])
        else:
            if not np.all(cell_vectors.shape == np.array([self.ndim, self.ndim])):
                raise ValueError("Cell repetition vector does not match with voxel data!")
            self.cell_vectors = cell_vectors.astype(np.int)
            if not np.all(cell_vectors == self.cell_vectors):
                raise ValueError("Cell repetition vector must be integers!")
            if not np.allclose(self.cell_vectors, np.triu(self.cell_vectors)):
                raise ValueError("Cell repetition vector must be upper triangular! %s" % self.cell_vectors)
        if pad:
            # make a periodic padding of 1px, useful for interpolation
            self.voxels = F.pad(torch.unsqueeze(self.voxels, 0), \
                                (0, 1) * self.ndim, mode='circular', value=0)[0]

    def wrap_indices(self, indices):              
        '''
        Wraps a (N * d) matrix of d-dimensional indices to inside the voxel grid.
        index positions does not have to be integers, as they can be inputs 
        for interpolation.
        Makes use of the triangular property of cell vector matrix. Wraps
        to inside the primitive cell from HIGHER to LOWER dimensions.
        '''
        for d in range(self.ndim - 1, -1, -1):
            
            if isinstance(indices, torch.Tensor):
                # pytorch uses torch.div() for true floordiv
                offset = torch.div(indices[:, d:d+1], self.cell_vectors[d, d],
                    rounding_mode="floor")
            else:
                # numpy integer division is floordiv
                offset = indices[:, d:d+1] // self.cell_vectors[d, d]
            indices -= offset * self.cell_vectors[:, d:d+1].T
        return indices

    def query(self, indices):
        '''
        Directly query a list of integer indices.
        '''
        shape = indices.shape
        if len(shape) > 2:
            indices = indices.view(-1, self.ndim)
        indices_3d = tuple(self.wrap_indices(indices).T)
        vals = torch.cat([self.voxels[i][indices_3d] \
                          for i in range(self.voxels.shape[0])], 0)
        if len(shape) > 2:
            vals = vals.view(self.voxels.shape[0], *shape[:-1])
        return vals

    def query_interp(self, indices):
        '''
        Query a list of float indices by interpolation
        The interpolator is set up the first time this method
        is called.
        '''
        if self.interpolator is None:
            gridpoints = [np.arange(self.voxels.shape[i + 1]) for i in range(self.ndim)]
            self.interpolator = [RegularGridInterpolator(
                gridpoints, self.voxels[i].numpy(), method='linear') for i in range(self.voxels.shape[0])]
        shape = indices.shape
        if len(shape) > 2:
            indices = indices.reshape((-1, self.ndim))
        indices_3d = self.wrap_indices(indices)
        vals = torch.cat([torch.from_numpy(interp(indices_3d)).float() for interp in self.interpolator], 0)
        if len(shape) > 2:
            vals = vals.view(self.voxels.shape[0], *shape[:-1])
        return vals


class CellGrid3(CellGrid):
    '''
    3D cell grid
    '''

    def __init__(self, voxels, cell_vectors=None, pad=True):
        if len(voxels.shape) != 4:
            raise ValueError("Grid must have 3 dimensions!")
        super().__init__(voxels, cell_vectors, pad)
        self.interpolator = None
        # determine if the voxel data needs to be transformed
        # by checking the shape. If so, the voxels are transformed 
        # into Cartesian space using the cell vectors
        if np.any(np.diag(cell_vectors) + 1 \
            != np.array(self.voxels.shape[1:])):
            self.back_transform(cell_vectors)
            
    def regular_grid(self, size):
        xs = np.arange(size)
        return torch.from_numpy(np.transpose(np.array(np.meshgrid(xs, xs, xs, indexing='ij')), (1, 2, 3, 0)))
    
    def unitcell_grid(self, cell_vectors=None, circular=True):
        cell_vectors = self.cell_vectors if cell_vectors is None else cell_vectors
        xs = np.arange(cell_vectors[0][0] + int(circular))
        ys = np.arange(cell_vectors[1][1] + int(circular))
        zs = np.arange(cell_vectors[2][2] + int(circular))
        return np.transpose(np.array(np.meshgrid(xs, ys, zs, indexing='ij')), (1, 2, 3, 0))
        
    def back_transform(self, cell_vectors):
        # no need to transform if cell vector matrix is diagonal
        if np.allclose(cell_vectors, np.diag(np.diag(cell_vectors))):
            return
        # needs to reset cell vectors for wrapping
        self.cell_vectors = np.diag(np.array(self.voxels.shape[1:]) - 1)
        pts = self.unitcell_grid(cell_vectors)
        pts_frac = np.matmul(pts.reshape(-1, 3), np.linalg.inv(cell_vectors.T))
        pts_frac *= (np.array(self.voxels.shape[1:]) - 1).reshape(1, 3)
        self.voxels = self.query_interp(pts_frac.reshape(pts.shape))
        self.interpolator = None
        self.cell_vectors = cell_vectors
    
    @property
    def cart_cell(self):
        '''
        Returns a Cartesian unit cell.
        '''
        return self.voxels[:, :-1, :-1, :-1]

    def sample_translate(self, n):
        '''
        Samples n random translation vectors in the unit cell.
        '''
        locs = []
        for i in range(self.ndim):
            locs.append(np.random.randint(0, self.voxels.shape[i + 1], n))
        return np.array(locs).T

    def sample_rotate(self, n):
        '''
        Samples n random rotation operations in the unit cell.
        '''
        rots = []
        for i in range(n):
            rots.append(get_rand_rotation())
        return rots

    def get_translate(self, size, translations):
        samples = []
        ntrans = len(translations)
        pts = np.tile(self.regular_grid(size).reshape(-1, 3), (ntrans, 1, 1))
        pts += np.expand_dims(translations, 1)
        samples = self.query(pts.reshape(-1, 3))
        samples = samples.view(self.voxels.shape[0], ntrans, size, size, size).permute(1, 0, 2, 3, 4)
        return samples

    def get_rotate(self, size, rotations, centers):
        samples = []
        pts = self.regular_grid(size).reshape(-1, 3)
        for r, c in zip(rotations, centers):
            samples.append(self.query_interp(
                apply_rotation(r, pts + c.reshape(1, 3))).reshape([size]*3))
        return samples

    def get_transpose(self, size, translations):
        samples = []
        pts = self.regular_grid(size)
        for t in translations:
            samples.append(self.query(
                np.transpose(pts + t, (1, 2, 0, 3)).reshape(-1, 3)
                ).reshape([size]*3))
            samples.append(self.query(
                np.transpose(pts + t, (2, 0, 1, 3)).reshape(-1, 3)
                ).reshape([size]*3))
        return np.array(samples)
   
    def symmetry_apply(self, transform):
        ind = self.unitcell_grid(circular=True).reshape(-1, 3)
        ind_sym = torch.round(torch.from_numpy(np.matmul(ind, transform[:, :3, :]) + transform[:, 3:4, :])).long()
        val_sym = self.query(ind_sym)
        val_mean = torch.mean(val_sym, 1).view(self.voxels.shape)
        return val_mean

    def get_positional_encoding(self, basis, transform):
        indices = self.unitcell_grid(circular=True) + 0.5
        grid_sizes = indices.shape[:-1]
        indices = np.matmul(indices.reshape(-1, 3), transform[:, :3, :]) + transform[:, 3:4, :]
        #indices = indices.reshape(1, -1, 3) / grid_sizes
        indices = np.dot(indices, basis.T)
        indices = np.transpose(indices, (0, 2, 1)).astype(np.float32).reshape(-1, basis.shape[0], *grid_sizes)
        #print(indices.shape)
        pos_enc = np.concatenate([
                np.max(np.sin(indices * 2 * np.pi), axis=0),
                np.max(np.cos(indices * 2 * np.pi), axis=0),
                ], axis=0)
        return torch.from_numpy(pos_enc).float()

    def get_symmetry_transforms(self, path, cartesian=True):
        return get_transforms(path, self.cell_vectors, cartesian=cartesian)

    def symmetrize(self, path, positional_encoding=False):
        '''
        Modifies the cell grid in place to make the voxels symmetric.
        '''
        tokens = get_symmetry_tokens(path)
        transform = self.symmetry_matrices(tokens)
        self.voxels = self.symmetry_apply(transform)
        if positional_encoding == "symmetrized":
            transform_frac = self.symmetry_matrices(tokens, cartesian=False)
            pos_enc = self.get_positional_encoding(pos_basis, transform_frac)
            return pos_enc
        elif positional_encoding == "base":
            transform_frac = self.symmetry_matrices([["+x", "+y", "+z"]], cartesian=False)
            pos_enc = self.get_positional_encoding(pos_basis, transform_frac)
            return pos_enc
        else:
            return self

def get_rand_rotation():
    u = np.random.random(3)
    coeffs = np.array([np.sqrt(1 - u[0])] * 2 + [u[0]] * 2)
    trigs = np.array([np.sin(2*np.pi*u[0]), np.cos(2*np.pi*u[1]), np.sin(2*np.pi*u[1]), np.cos(2*np.pi*u[1])])
    return Rotation.from_quat(coeffs * trigs)

def apply_rotation(r, pts):
    '''
    Applies rotation R to points PTS around the 
    centroid of PTS.
    '''
    mean = np.mean(pts, axis=0)
    return r.apply(pts - mean) + mean

def pool(voxels, npool, mean=True):
    from skimage.measure import block_reduce
    newshape = np.array(voxels.shape) // npool
    newvoxels = block_reduce(voxels, block_size=(npool,) * len(voxels.shape), func=np.mean if mean else np.sum)
    newvoxels = newvoxels[tuple(slice(0, x) for x in newshape)]
    return newvoxels


def get_symmetry_tokens(path):
    symmetry_tokens = []
    with open(path, "r") as f:
        lines = f.read().splitlines() 
        i = 1
        while not lines[i - 1].strip() in ["_space_group_symop_operation_xyz",
            "_symmetry_equiv_pos_as_xyz"]:
            i += 1
        while lines[i] and lines[i] != 'loop_' and lines[i][0] != '_':
            symmetry_tokens.append(lines[i].strip("'").split(","))
            i += 1
    return symmetry_tokens

def get_transforms(path, cell_vectors, cartesian=True):
    tokens = get_symmetry_tokens(path)
    n_ops = len(tokens)
    # each COLUMN is a linear transformation
    # because we operate on indices in row vectors
    transform = np.zeros((n_ops, 4, 3)) 
    for i, t in enumerate(tokens):
        for d in range(3):
            coeffs = parse_symm_op(t[d])
            transform[i, :, d] = coeffs
    # Fractional coordinates to Cartesian coordinates
    M = cell_vectors
    # NOTE: fractional coordinate transforms are returned
    # when cartesian=None
    if cartesian is True:
        transform[:, :3, :] = np.linalg.inv(M.T) @ transform[:, :3, :] @ M.T
        transform[:, 3:4, :] = transform[:, 3:4, :] @ M.T
    elif cartesian is False:
        transform[:, :3, :] = np.linalg.inv(M.T) @ transform[:, :3, :]
    return transform

def parse_symm_op(token):
    coeffs = [0, 0, 0, 0] # homogeneous coordinates
    neg = False
    number = ""
    for c in token:
        if c == '-' or c == '+':
            neg = c == '-'
            if number:
                coeffs[3] += eval(number) # only works in Python3!
                number = ""
        elif c == 'x':
            coeffs[0] = -1 if neg else 1
        elif c == 'y':
            coeffs[1] = -1 if neg else 1
        elif c == 'z':
            coeffs[2] = -1 if neg else 1
        elif number or 49 <= ord(c) <= 57: # numbers 1..9
            number += c
    if number:
        coeffs[3] += eval(number)
    return coeffs

def calculate_cell_vectors(cell_info, pool):
    spacing = np.array(cell_info['ortho_length']) / np.array(cell_info['grid_sizes'])
    grid_trans = np.round(cell_info['translation_vector'] / spacing).astype(np.int).T
    cell_vectors = grid_trans // pool
    return cell_vectors

pos_basis = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [2, 0, 0],
    [0, 2, 0],
    [0, 0, 2],
    [4, 0, 0],
    [0, 4, 0],
    [0, 0, 4]
])

