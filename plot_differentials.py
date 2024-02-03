import taichi as ti
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from dataset import create_dataset
from model.unet import Symmetrize
from utils import TraPPE, get_ztb, get_transforms, calculate_cell_vectors
from utils.isotherms import *
from utils.visualization import *


if __name__ == "__main__":

    
    molecules = ["CO2", "H2S", "i-butane", "n-butane", "Xe", "Kr"]
    keys = ["IRR-0", "MFI-0"]
    funcs = {"langmuir": solve_langmuir, "quadratic": solve_quadratic}
    
    for molecule in molecules:
        for pool in [2]:#[1, 2, 4]:
            ds = create_dataset(molecule, pool)
            for key in keys:
                grid_info, egrid, ncell = get_ztb(ds.ztb_path, key, ds.atoms)
                info, voxels_all = ds.get_all_trajctories(key, ncell)
                cell_vectors = calculate_cell_vectors(info, ds.pool)
                transforms = get_transforms(ds.cif_path + "/%s.cif" % key, cell_vectors, cartesian=None)
                for name, func in funcs.items():
                    local_map, global_params = func(ds.log_p, ds.inv_t, voxels_all, ds.pool)
                    print(local_map.shape)
                    local_map = Symmetrize.apply(
                        local_map.unsqueeze(0),
                        torch.tensor(cell_vectors).contiguous(),
                        torch.tensor(transforms).contiguous()
                    ).squeeze(0)
                    combined_img = generate_combined_img(local_map).numpy()
                    plt.imshow(combined_img, cmap="RdBu", norm=plt.Normalize(-5, 5))
                    savefile = f"results/analysis/{molecule}-{key}-{name}-{0.1 * ds.pool}.png"
                    plt.savefig(savefile, format="png")
                    print("Saved:", savefile)
