import os

import matplotlib.pyplot as plt
import numpy as np
import torch

rdbu = plt.get_cmap("RdBu")
viridis = plt.get_cmap("viridis")
x_norm = plt.Normalize(-1, 1)
y_norm = plt.Normalize(-1, 1)

def pad_to_cube(x, l_max):
    x_padded = torch.zeros(x.shape[0], l_max, l_max, l_max)
    x_padded[
        :,
        (l_max - x.shape[1]) // 2 : (l_max - x.shape[1]) // 2 + x.shape[1],
        (l_max - x.shape[2]) // 2 : (l_max - x.shape[2]) // 2 + x.shape[2],
        (l_max - x.shape[3]) // 2 : (l_max - x.shape[3]) // 2 + x.shape[3]
    ] = x
    return x_padded

def log_predicted_images(trainer, x, y_true, y_pred):
    if not (x.shape[1] == x.shape[2] == x.shape[3]):
        l_max = max(x.shape[1], x.shape[2], x.shape[3])
        x, y_true, y_pred = pad_to_cube(x, l_max), pad_to_cube(y_true, l_max), pad_to_cube(y_pred, l_max)
    if x.shape[0] < 2:
        x = torch.cat([x, torch.zeros(2 - x.shape[0], x.shape[1], x.shape[2], x.shape[3])], 0)
    if x.shape[1] != y_true.shape[1]:
        crop = (x.shape[1] - y_true.shape[1]) // 2
        x = x[:, crop:-crop, crop:-crop, crop:-crop]
   
    # projections
    x_proj = torch.stack((x.mean(1), x.mean(2), x.mean(3)), 2)
    y_true_proj = torch.stack((y_true.mean(1), y_true.mean(2), y_true.mean(3)), 2)
    y_pred_proj = torch.stack((y_pred.mean(1), y_pred.mean(2), y_pred.mean(3)), 2)
    # combine image
    x_combined = x_proj.permute(1, 0, 2, 3).flatten(1, 3)
    y_true_combined = y_true_proj.flatten(2, 3).flatten(0, 1)
    y_pred_combined = y_pred_proj.flatten(2, 3).flatten(0, 1)
    # apply colormap
    x_combined = viridis(x_norm(x_combined.cpu().numpy()))
    y_combined = rdbu(y_norm(torch.cat((y_true_combined, y_pred_combined), 1).cpu().numpy()))
    img_combined = np.concatenate((x_combined, y_combined), 0)
    save_path = os.path.join(trainer.default_root_dir, "artifacts", f"step{trainer.global_step}.png")
    plt.imsave(save_path, img_combined)
    trainer.logger.experiment.log_artifact(trainer.logger.run_id, save_path)