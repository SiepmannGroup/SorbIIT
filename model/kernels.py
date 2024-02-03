import torch
import taichi as ti

DIM = 3

ti.init(arch=ti.cpu)

def wrap_indices(indices, cell_vectors):     
    for d in range(DIM - 1, -1, -1):
        if isinstance(indices, torch.Tensor):
            # pytorch uses torch.div() for true floordiv
            offset = torch.div(indices[:, d:d+1], cell_vectors[d, d],
                rounding_mode="floor")
        else:
            # numpy integer division is floordiv
            offset = indices[:, d:d+1] // cell_vectors[d, d]
        indices -= offset.long() * cell_vectors[:, d:d+1].T.long()
    return indices

def query_indices(voxels, indices, cell_vectors):
    '''
    Directly query a list of integer indices.
    '''
    shape = indices.shape
    if len(shape) > 2:
        indices = indices.view(-1, DIM)
    indices_3d = tuple(wrap_indices(indices, cell_vectors).T)
    vals = torch.cat([voxels[i][indices_3d] \
                        for i in range(voxels.shape[0])], 0)
    if len(shape) > 2:
        vals = vals.view(voxels.shape[0], *shape[:-1])
    return vals

def symmetry_apply(voxels, unit_cell_indices, cell_vectors, transform):
    ind = unit_cell_indices.reshape(-1, 3)
    transform[:, :3, :] = torch.inverse(cell_vectors.T) @ transform[:, :3, :] @ cell_vectors.T
    transform[:, 3:4, :] = transform[:, 3:4, :] @ cell_vectors.T
    ind_sym = torch.round(ind @ transform[:, :3, :] + transform[:, 3:4, :]).long()
    val_sym = query_indices(voxels, ind_sym, cell_vectors)
    val_mean = torch.mean(val_sym, 1).view(voxels.shape)
    return val_mean


@ti.kernel
def symmetrize_kernel(x: ti.any_arr(), y: ti.any_arr(), cell_vectors: ti.any_arr(), transforms: ti.any_arr()):
    cell_matrix = ti.Matrix([
        [cell_vectors[0, 0], cell_vectors[0, 1], cell_vectors[0, 2]],
        [cell_vectors[1, 0], cell_vectors[1, 1], cell_vectors[1, 2]],
        [cell_vectors[2, 0], cell_vectors[2, 1], cell_vectors[2, 2]],
    ])
    for I in ti.grouped(x):
        #y[I] = float("-inf")
        for i in range(transforms.shape[0]):
            affine = ti.Matrix([
                [transforms[i, 0, 0], transforms[i, 0, 1], transforms[i, 0, 2]],
                [transforms[i, 1, 0], transforms[i, 1, 1], transforms[i, 1, 2]],
                [transforms[i, 2, 0], transforms[i, 2, 1], transforms[i, 2, 2]],
            ])
            translate = ti.Vector([
                transforms[i, 3, 0],
                transforms[i, 3, 1], 
                transforms[i, 3, 2],
            ])
            # fractional coordinates
            idx = ti.Vector([I[2], I[3], I[4]])
            p = cell_matrix.inverse() @ (idx + 0.5)
            p_symm = affine.transpose() @ p + translate
            # convert to index and wrap to Cartesian box
            idx_new = wrap_indices_single(int(cell_matrix @ p_symm), cell_vectors)
            #if any(idx_new != idx):
            #    print(idx_new, idx, p, p_symm)
            y[I] = y[I] + x[I[0], I[1], idx_new.x, idx_new.y, idx_new.z] / transforms.shape[0]
            #y[I] = max(y[I], x[I[0], I[1], idx_new.x, idx_new.y, idx_new.z])
            


def pad(x, cell_vectors, pad_amount):
    if isinstance(pad_amount, int):
        pad_amount = (pad_amount,) * 3
    y = torch.empty((
        x.shape[0],
        x.shape[1],
        x.shape[2] + 2 * pad_amount[0],
        x.shape[3] + 2 * pad_amount[1],
        x.shape[4] + 2 * pad_amount[2],
    ), dtype=torch.float32, device=x.device)
    y[
        :,
        :,
        pad_amount[0] : pad_amount[0] + x.shape[2],
        pad_amount[1] : pad_amount[1] + x.shape[3],
        pad_amount[2] : pad_amount[2] + x.shape[4]
    ] = x
    pad_kernel(y, cell_vectors, *pad_amount)
    return y

@ti.func
def wrap_indices_single(indices: ti.template(), cell_vectors: ti.template()):     
    indices_new = ti.Vector([indices[0], indices[1], indices[2]])
    for d in ti.static(range(DIM - 1, -1, -1)):
        offset = indices_new[d] // int(cell_vectors[d, d])
        indices_new = indices_new - int(offset * ti.Vector([
            cell_vectors[0, d],
            cell_vectors[1, d],
            cell_vectors[2, d],
        ]))
    #if indices_new.x < 0 or indices_new.y < 0 or indices_new.z < 0 \
    #    or indices_new.x >= cell_vectors[0, 0] or indices_new.y >= cell_vectors[1, 1] or indices_new.z >= cell_vectors[2, 2]:
    #    print(indices, indices_new, offset_all, cell_vectors[0, 0], cell_vectors[1, 1], cell_vectors[2, 2])
    return indices_new

@ti.kernel
def pad_kernel(y: ti.any_arr(), cell_vectors: ti.any_arr(), pw: int, ph: int, pd: int):
    """
    Shapes:
        x: (b * c * w * h * d)
        y: (b * c * (w + 2 * pw) * (h + 2 * ph) * (d + 2 * pd))
        cell_vectors: (b * 3 * 3)
    """
    w, h, d = y.shape[2] - 2 * pw, y.shape[3] - 2 * ph, y.shape[4] - 2 * pd
    n_pad_d = (w + 2 * pw) * (h + 2 * ph) * 2 * pd
    n_pad_h = (w + 2 * pw) * d * 2 * ph
    n_pad_w = h * d * 2 * pw
    pad_vector = ti.Vector([pw, ph, pd])
    for b, idx in ti.ndrange(y.shape[0], n_pad_d + n_pad_h + n_pad_w):
        i, j, k = 0, 0, 0
        if idx < n_pad_d:
            # depth padding
            i = idx // (2 * pd) // (h + 2 * ph)
            j = idx // (2 * pd) % (h + 2 * ph)
            k = idx % (2 * pd)
            k = k + d if k >= pd else k
        elif idx < n_pad_d + n_pad_h:
            # height padding:
            i = (idx - n_pad_d) // (2 * ph) // d
            j = (idx - n_pad_d) % (2 * ph)
            j = j + h if j >= ph else j
            k = (idx - n_pad_d) // (2 * ph) % d + pd
        else: 
            # width padding
            i = (idx - n_pad_d - n_pad_h) % (2 * pw)
            i = i + w if i >= pw else i
            j = (idx - n_pad_d - n_pad_h) // (2 * pw) // d + ph
            k = (idx - n_pad_d - n_pad_h) // (2 * pw) % d + pd
        indices = ti.Vector([i, j, k])
        indices_new = wrap_indices_single(indices - pad_vector, cell_vectors) + pad_vector
        # pad by each channel
        for c in range(y.shape[1]):
            y[b, c, i, j, k] = y[b, c, indices_new.x, indices_new.y, indices_new.z]

if __name__ == "__main__":
    from datasets import SpatialDerivativeDataset
    from ztbutils import TraPPE
    from visualization import visualize_slice

    def symmetrize(x, cell_vectors, transforms):
        y = torch.zeros_like(x)
        symmetrize_kernel(x, y, cell_vectors, transforms)
        return y

    atoms = [TraPPE.C_CO2, TraPPE.O_CO2]
    dataset = SpatialDerivativeDataset(atoms, 4, (64, 64), 
                "../vision-data/CO2-highp.h5",
                "/home/andrewsun/ssd0/IZASC/ff/",
                [1, 3.2, 10, 32],
                [256, 270, 286, 303, 323, 343, 370, 400],
                    ntrans=6, nrot=0, axisrot=True,
            cif_path="/home/andrewsun/ssd0/IZASC/cif/",
            positional_encoding=None,
            normalize_all=True,
            return_symmetrize_func=True)
    zeolite = "AWW-0"
    j = dataset.keys.tolist().index(zeolite)
    x, y, symm_info = dataset.get_unit_cells(j)
    cell_vectors = symm_info[1].contiguous()
    transforms = symm_info[2]
    x += torch.rand_like(x) * 0.02
    x = x.contiguous()
    new_size = (30, 50, 35)
    print(transforms)
    print(torch.diagonal(cell_vectors), x.shape)
    x = torch.nn.functional.interpolate(
                    x, new_size, mode='trilinear', align_corners=False
                )
    cell_vector_scale = torch.tensor(new_size) / torch.diagonal(cell_vectors)
    cell_vector_new = cell_vectors.float() * cell_vector_scale.unsqueeze(1)
    print(cell_vector_new)
    x = symmetrize(x, cell_vector_new, transforms)
    #x = symmetry_apply(x.squeeze(), symm_info[0], cell_vector_new, transforms).unsqueeze(0)
    y = pad(x, cell_vector_new, (10, 10, 0))
    print(x.shape)
    print(y.shape)
    visualize_slice(y.squeeze(0), cmap="RdBu",)
