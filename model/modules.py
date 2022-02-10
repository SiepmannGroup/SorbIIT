import torch
import pytorch_lightning as pl

from .logs import log_predicted_images
from .unet import Symmetrize

EPS = 1e-8



class CellUNetModule(pl.LightningModule):
    def __init__(
        self,
        n_in,
        n_out,
        generator_class,
        generator_config,
        error_loss="mse",
        learning_rate=1e-3,
        betas=(0.9, 0.99),
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = generator_class(n_in, n_out, **generator_config)

        if error_loss == "l1":
            self.err_loss_func = torch.nn.L1Loss()
        elif error_loss == "mse":
            self.err_loss_func = torch.nn.MSELoss()
        else:
            raise NotImplementedError

        # cache current results
        self.y_pred = None

    def forward(self, x, symm_info):
        _, cell_vectors, transforms = symm_info
        self.y_pred = self.generator(x, cell_vectors.cpu(), transforms.cpu())
        return self.y_pred
    
    def step(self, x, y_true, symm_info):
        loss_dict = {}
        y_pred = self(x, symm_info)
        loss_dict["err"] = self.err_loss_func(y_pred, y_true)
        return loss_dict
        
    def training_step(self, batch, batch_idx):
        x, y_true, symm_info = batch
        y_true = Symmetrize.apply(y_true, symm_info[1].cpu(), symm_info[2].cpu())
        loss_dict = self.step(x, y_true, symm_info)
        for k, v in loss_dict.items():
            self.log(k, v, prog_bar=True)
        loss = sum(loss_dict.values())
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true, symm_info = batch
        y_true = Symmetrize.apply(y_true, symm_info[1].cpu(), symm_info[2].cpu())
        loss_dict = self.step(x, y_true, symm_info)
        for k, v in loss_dict.items():
            self.log("val_" + k, v)
        loss = sum(loss_dict.values())
        self.log("val_loss", loss, prog_bar=True)
        log_predicted_images(self.trainer, x[0, :self.hparams.n_in], y_true[0], self.y_pred[0])
        return loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        b1 = self.hparams.betas[0]
        b2 = self.hparams.betas[1]

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        return opt_g
