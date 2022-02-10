import argparse
from datetime import datetime
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger


from model import CellUNetModule, CellUNet
from dataset import ZTBDatasetModule
from utils import TraPPE

dataset = ZTBDatasetModule(
    train_size=120,
    val_size=5,
    atoms=[TraPPE.C_CO2, TraPPE.O_CO2],
    pool_level=2,
    window_size=(64, 64),
    traj_path="/mnt/ssd1/andrew/vision-data/CO2-highp.h5",
    ztb_path="/mnt/ssd0/andrew/IZASC/ff/",
    pressures=[1, 3.2, 10, 32],
    temperatures=[256, 270, 286, 303, 323, 343, 370, 400],
    ntrans=0,
    nrot=0,
    axisrot=True,
    cif_path="/mnt/ssd0/andrew/IZASC/cif/",
    positional_encoding=None,
    normalize_all=True,
    return_symmetrize_func=True,
    symmetrize_in_place=False,
)

in_shape, out_shape = dataset.get_shapes()
n_in = 2
n_pos = in_shape[1] - n_in
n_out = out_shape[1]

generator_config = {   
    "channels": (16, 32, 64, 128, 256),
}


model = CellUNetModule(
    n_in,
    n_out,
    CellUNet,
    generator_config,
    adversarial_loss="bce",
    error_loss="mse",
    learning_rate=1e-3,
    betas=(0.5, 0.99),
)


run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(os.path.join("models", run_name, "artifacts"))
os.makedirs(os.path.join("models", run_name, "checkpoints"))

save_callback = pl.callbacks.ModelCheckpoint(
    monitor="val_loss",
    save_top_k=20,
    every_n_epochs=5,
    dirpath=os.path.join("models", run_name, "checkpoints"),
    save_last=True,
)

logger = MLFlowLogger(
    experiment_name="ztbgan-cellunet-regress",
    tracking_uri="http://localhost:5000",
    run_name=run_name,
)


trainer = pl.Trainer(
    default_root_dir=os.path.join(os.getcwd(), "models", run_name),
    weights_save_path=os.path.join(os.getcwd(), "models", run_name, "checkpoints"),
    gpus=1,#[1],
    accumulate_grad_batches=4,
    precision=32,
    callbacks=[save_callback],
    logger=logger,
    val_check_interval=25,
)

trainer.fit(model, dataset)
