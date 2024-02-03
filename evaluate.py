
from matplotlib import projections
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from model import CellUNetModule, CellUNet, Symmetrize
from dataset import create_dataset
from utils import TraPPE, get_ztb
from utils.isotherms import *
from utils.visualization import *


dataset = create_dataset("CO2", pool_level=2)

y_ext_mean = -20.0
y_ext_std = 8.0
y_int_std = 30.0

#ckpt_file = "../models/20220325-223840/checkpoints/epoch=214-step=6444.ckpt"
#ckpt_file = "../models/20220325-223840/checkpoints/epoch=149-step=4494.ckpt"
ckpt_file = "../models/20220325-223840/checkpoints/epoch=69-step=2094.ckpt"

model = CellUNetModule.load_from_checkpoint(ckpt_file)
model.eval()

key = "IRR-0"

idx = dataset.keys.tolist().index(key)
x, y_true, symm_info = dataset[idx]
y_pred = model(x, symm_info).detach()
y_true = Symmetrize.apply(y_true, symm_info[1].cpu(), symm_info[2].cpu())

mask_true = (y_true[:, -2:-1] ** 2) > 1e-10
mask_true = torch.tile((y_true[:, -1] ** 2) > 0, (1, y_true.shape[1], 1, 1, 1)).bool()
mask_pred = torch.tile(y_pred[:, 0] > np.log(0.99/0.01), (1, y_true.shape[1], 1, 1, 1)).bool()

accuracy = (mask_pred == mask_true).float().mean()
print("Accuracy:", accuracy)
iou = (mask_pred & mask_true).sum() / (mask_pred | mask_true).sum()
print("IoU:", iou)

y_true_masked = y_true[mask_true]
y_pred_masked = y_pred[:, 1:][mask_true]

y_pred = torch.masked_fill(y_pred[:, 1:], ~mask_pred, 0)[0]
print(y_pred.mean((1, 2, 3)))

y_pred_vis = y_pred.clone()
y_true_vis = torch.masked_fill(y_true, ~mask_true, 0)[0]
y_true_vis[0] /= 3
y_true_vis[1:3] *= 3
y_pred_vis[0] /= 3
y_pred_vis[1:3] *= 3
visualize_projection(x[0], y_true_vis, y_pred_vis, f"results/{key}.png")

grid_info, _, ncell = get_ztb(dataset.ztb_path, key, dataset.atoms)
traj_info, voxels_all = dataset.get_all_trajctories(key, ncell)
raw, popt, _ = solve_quadratic(dataset.log_p, dataset.inv_t, voxels_all, dataset.pool)

print(torch.std(raw, dim=(1, 2, 3)))
print(torch.std(y_true[0], dim=(1, 2, 3)))
print(torch.std(y_pred, dim=(1, 2, 3)))

y_pred[1:] *= dataset.y_int_std
print(y_pred.shape)
voxels_pred = predict_quadratic(dataset.log_p, dataset.inv_t, y_pred, popt, dataset.pool)
print(voxels_pred.shape)


voxels_pred = voxels_pred.view(len(dataset.log_p), len(dataset.inv_t), *voxels_pred.shape[1:])
voxels_all /= 0.1 * dataset.pool ** 3
voxels_pred /= 0.1 * dataset.pool ** 3


projection = voxels_pred.mean(-1).transpose(1, 2).flatten(0, 1).flatten(1, 2)
print(projection.shape)
plt.imshow(projection.numpy(), cmap="viridis", norm=plt.Normalize(0, 0.001))
savefile = f"results/full-loading-surface/{key}-pred.png"
plt.savefig(savefile, format="png")
print("Saved:", savefile)

projection = voxels_all.view(voxels_pred.shape).mean(-1).transpose(1, 2).flatten(0, 1).flatten(1, 2)
plt.imshow(projection.numpy(), cmap="viridis", norm=plt.Normalize(0, 0.001))
savefile = f"results/full-loading-surface/{key}-true.png"
plt.savefig(savefile, format="png")
print("Saved:", savefile)