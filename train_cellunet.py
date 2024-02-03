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
    molecule="CO2",
    pool_level=2,
    solver="quadratic",
    num_workers=4,
)

in_shape, out_shape = dataset.get_shapes()
n_in = len(dataset.atoms)
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
    l1_weight=1,
    l2_weight=0,
    bce_weight=1,
    learning_rate=1e-3,
    betas=(0.5, 0.99),
    elu=True,
    reg_weight=0.1,
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
    gpus=1,
    accumulate_grad_batches=4,
    precision=32,
    callbacks=[save_callback],
    logger=logger,
    val_check_interval=25,
)
print(model.summarize(mode="full"))
trainer.fit(model, dataset)
