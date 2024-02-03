import taichi as ti
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import r2_score

from model import CellUNetModule, CellUNet
from dataset import create_dataset
from model import unet
from utils import TraPPE, get_ztb
from utils.isotherms import *

molecule = "CO2"
ds = create_dataset(molecule, 2)

funcs = {"langmuir": (solve_langmuir, predict_langmuir), "quadratic": (solve_quadratic, predict_quadratic)}

columns = [
    "Zeolite",
    "Total MAE Langmuir", "Total R2 Langmuir",
    "Local MAE Langmuir", "Local R2 Langmuir",
    "Local Mean Langmuir dQ",
    "Local Mean Langmuir ds", "Local Mean Langmuir dh",
    "Total MAE Quadratic", "Total R2 Quadratic",
    "Local MAE Quadratic", "Local R2 Quadratic",
    "Local Mean Quadratic dQ",
    "Local Mean Quadratic ds1", "Local Mean Quadratic dh1",
    "Local Mean Quadratic ds2", "Local Mean Quadratic dh2",
]
rows = []

try:
    df_old = pd.read_csv("results/isotherm-fit.csv", index_col=0)
except FileNotFoundError:
    df_old = pd.DataFrame([], columns=columns)


for i in tqdm(range(len(ds.keys))):
    key = str(ds.keys[i])
    if key in df_old["Zeolite"].astype(str).tolist():
        print(f"Skipping {key}, already done")
        continue
    try:
        grid_info, egrid, ncell = get_ztb(ds.ztb_path, key, ds.atoms)
        del egrid
        if np.prod(grid_info["cell_parameters"]) > 20000:
            print(f"Skipping {key}, framework too large")
        else:
            grid_info, egrid, ncell = get_ztb(ds.ztb_path, key, ds.atoms)
            info, voxels_all = ds.get_all_trajctories(key, ncell)
            row = [key]
            for key, value in funcs.items():
                solve, predict = value
                local_map, popt, tot, tot_pred = solve(ds.log_p, ds.inv_t, voxels_all, ds.pool, return_total=True)
                tot_mae = np.abs(tot - tot_pred).mean()
                tot_r2 = r2_score(tot, tot_pred)
                voxels_pred = predict(ds.log_p, ds.inv_t, local_map, popt, ds.pool)
                loc_mae = torch.abs(voxels_all.squeeze() - voxels_pred).mean().item()
                loc_r2 = r2_score(voxels_all.numpy().ravel(), voxels_pred.numpy().ravel())
                row += [tot_mae, tot_r2, loc_mae, loc_r2]
                row += local_map.mean((1, 2, 3)).numpy().tolist()
            rows.append(row)
            df = pd.DataFrame(rows, columns=columns)
            df = pd.concat([df_old, df]).reset_index(drop=True)
            df.to_csv("results/isotherm-fit.csv")
    except KeyError as e:
        print(f"{type(e)}: {e}")
