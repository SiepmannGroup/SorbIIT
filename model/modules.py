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
        bce_weight=0,
        l1_weight=0,
        l2_weight=1,
        learning_rate=1e-3,
        betas=(0.9, 0.99),
        elu=False,
        reg_weight=0.1,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = generator_class(n_in, n_out if bce_weight == 0 else n_out + 1, **generator_config)

        if l1_weight == 0 and l2_weight == 0:
            print("Both loss weights are specified as zero. L2 weight will be reset to 1.")
            l2_weight = 1
        self.l1_loss = torch.nn.L1Loss()
        self.l1_weight = l1_weight
        self.l2_loss = torch.nn.MSELoss()
        self.l2_weight = l2_weight
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

        self.elu = elu
        self.reg_weight = reg_weight
        # cache current results
        self.y_pred = None

    def forward(self, x, symm_info):
        _, cell_vectors, transforms = symm_info
        self.y_pred = self.generator(x, cell_vectors.cpu(), transforms.cpu())
        if self.elu:
            self.y_pred = torch.cat((
                self.y_pred[:, 0:1],
                torch.nn.functional.elu(self.y_pred[:, 1:2]) + 1,
                self.y_pred[:, 2:]),
                1)
        if self.reg_weight < 0:
            logit, values = self.y_pred[:, 0:1], self.y_pred[:, 1:]
            values = torch.masked_fill(values, logit < 0, 0.0)
            self.y_pred = torch.cat((
                logit,
                values[:, 0:1] / values[:, 0:1].mean((2, 3, 4), keepdim=True),
                values[:, 1:] - values[:, 1:].mean((2, 3, 4), keepdim=True),
            ), 1)
        return self.y_pred
    
    def step(self, x, y_true, symm_info):
        loss_dict = {}
        y_pred = self(x, symm_info)
        print(y_true.flatten(2, 4).max(2)[0].squeeze())
        if self.bce_weight > 0:
            logit_pred = y_pred[:, 0]
            mask_true = ((y_true[:, -1] ** 2) > EPS).float()
            loss_dict["bce"] = self.bce_weight * self.bce_loss(logit_pred, mask_true)
            mask_true = mask_true.bool().squeeze()
            y_pred = y_pred[:, 1:, mask_true]
            y_true = y_true[:, :, mask_true]
        if self.l2_weight > 0:
            loss_dict["l2"] = self.l2_weight * self.l2_loss(y_pred, y_true)
        if self.l1_weight > 0:
            loss_dict["l1"] = self.l1_weight * self.l1_loss(y_pred, y_true)
        if self.reg_weight > 0:
            val_mean = y_pred.sum(2) / torch.prod(torch.tensor(y_true.shape[2:]))
            loss_dict["reg_extensive"] = ((val_mean[:, 0] - 1) ** 2).mean()
            loss_dict["reg_intensive"] = (val_mean[:, 1:] ** 2).mean()
        return loss_dict
        
    def training_step(self, batch, batch_idx):
        x, y_true, symm_info = batch
        y_true = Symmetrize.apply(y_true, symm_info[1].cpu(), symm_info[2].cpu())
        loss_dict = self.step(x, y_true, symm_info)
        for k, v in loss_dict.items():
            self.log(k, v, prog_bar=True)
        loss = sum(loss_dict.values())
        self.log("running_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true, symm_info = batch
        y_true = Symmetrize.apply(y_true, symm_info[1].cpu(), symm_info[2].cpu())
        loss_dict = self.step(x, y_true, symm_info)
        for k, v in loss_dict.items():
            self.log("val_" + k, v)
        loss = sum(loss_dict.values())
        self.log("val_loss", loss, prog_bar=True)
        mask_true = (y_true[:, -2:-1] ** 2) < EPS
        mask_pred = self.y_pred[:, 0:1] < 0
        y_true_vis = torch.cat([
            (y_true[0, -2:-1] ** 2) > EPS,
            torch.masked_fill(y_true, mask_true, 0)[0]
        ], 0)
        y_pred_vis = torch.cat([
            torch.sigmoid(self.y_pred[0, 0:1]),
            torch.masked_fill(self.y_pred[:, 1:], mask_pred, 0)[0]
        ], 0)
        accuracy = (mask_true == mask_pred).float().mean()
        self.log("accessible_accuracy", accuracy)
        iou = (mask_pred & mask_true).sum() / (mask_pred | mask_true).sum()
        self.log("accessible_iou", iou)
        sign_accuracy = (torch.sign(self.y_pred[:, 2:]) == torch.sign(y_true[:, 1:])).float().mean()
        self.log("sign_accuracy", sign_accuracy)
        log_predicted_images(self.trainer, x[0, :self.hparams.n_in], y_true_vis, y_pred_vis)
        return loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        b1 = self.hparams.betas[0]
        b2 = self.hparams.betas[1]

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        return opt_g


class CellUNetGANModule(pl.LightningModule):
    def __init__(
        self,
        n_in,
        n_out,
        generator_class,
        discriminator_class,
        generator_config,
        discriminator_config,
        adversarial_loss="bce",
        bce_weight=1,
        l1_weight=1,
        l2_weight=0,
        learning_rate=1e-3,
        betas=(0.9, 0.99),
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        super().__init__()
        self.save_hyperparameters()

        self.generator = generator_class(n_in, n_out if bce_weight == 0 else n_out + 1, **generator_config)
        self.discriminator = discriminator_class(n_in + n_out, **discriminator_config)

        if adversarial_loss == "bce":
            self.gan_loss_func = torch.nn.BCEWithLogitsLoss()
        elif adversarial_loss == "mse":
            self.gan_loss_func = torch.nn.MSELoss()
        else:
            raise NotImplementedError

        self.l1_loss = torch.nn.L1Loss()
        self.l1_weight = l1_weight
        self.l2_loss = torch.nn.MSELoss()
        self.l2_weight = l2_weight
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

        # cache current results
        self.y_pred = None

    def forward(self, x, symm_info):
        _, cell_vectors, transforms = symm_info
        self.y_pred = self.generator(x, cell_vectors.cpu(), transforms.cpu())
        if self.elu:
            self.y_pred = torch.cat((
                self.y_pred[:, 0:1],
                torch.nn.functional.elu(self.y_pred[:, 1:2]) + 1,
                self.y_pred[:, 2:]),
                1)
        if self.reg_weight < 0:
            logit, values = self.y_pred[:, 0:1], self.y_pred[:, 1:]
            values = torch.masked_fill(values, logit < 0, 0.0)
            self.y_pred = torch.cat((
                logit,
                values[:, 0:1] / values[:, 0:1].mean((2, 3, 4), keepdim=True),
                values[:, 1:] - values[:, 1:].mean((2, 3, 4), keepdim=True),
            ), 1)
        return self.y_pred
    

    def adversarial_loss(self, x, y, label):
        if x.shape[-1] != y.shape[-1]:
            crop = (x.shape[-1] - y.shape[-1]) // 2
            x = x[:, :, crop:-crop, crop:-crop, crop:-crop]
        pair = torch.cat([x, y], 1)
        prob_pred = self.discriminator(pair)
        label = torch.tensor(label).expand_as(prob_pred).float().to(x.device)
        return self.gan_loss_func(prob_pred, label)

    def generator_step(self, x, y_true, symm_info):
        loss_dict = {}
        y_pred = self(x, symm_info)
        if self.bce_weight > 0:
            logit_pred = y_pred[:, 0]
            mask_true = ((y_true[:, -1] ** 2) > EPS).float()
            loss_dict["bce"] = self.bce_weight * self.bce_loss(logit_pred, mask_true)
            mask_true = torch.tile(mask_true, (1, y_pred.shape[1] - 1, 1, 1, 1)).bool()
            y_pred_valid = y_pred[:, 1:][mask_true]
            y_true_valid = y_true[mask_true]
            y_pred = torch.masked_fill(y_pred[:, 1:], ~mask_true, 0)
        else:
            y_pred_valid = y_pred
            y_true_valid = y_true
        if self.l2_weight > 0:
            loss_dict["l2"] = self.l2_weight * self.l2_loss(y_pred_valid, y_true_valid)
        if self.l1_weight > 0:
            loss_dict["l1"] = self.l1_weight * self.l1_loss(y_pred_valid, y_true_valid)
        loss_dict["loss_g_fake"] = self.adversarial_loss(x, y_pred, 1)
        return loss_dict

    def discriminator_step(self, x, y_true, symm_info):
        loss_dict = {}
        y_pred = self(x, symm_info)
        if self.bce_weight > 0:
            mask_true = torch.tile((y_true[:, -1] ** 2) > EPS, (1, y_true.shape[1], 1, 1, 1)).bool()
            y_pred = torch.masked_fill(y_pred[:, 1:], ~mask_true, 0)
            y_true = torch.masked_fill(y_true, ~mask_true, 0)
        loss_dict["loss_d_real"] = self.adversarial_loss(x, y_true, 1) / 2
        loss_dict["loss_d_fake"] = self.adversarial_loss(x, y_pred, 0) / 2
        return loss_dict
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y_true, symm_info = batch
        y_true = Symmetrize.apply(y_true, symm_info[1].cpu(), symm_info[2].cpu())
        # train generator
        if optimizer_idx == 0:
            loss_dict = self.generator_step(x, y_true, symm_info)
        # train discriminator
        elif optimizer_idx == 1:
            loss_dict = self.discriminator_step(x, y_true, symm_info)
        for k, v in loss_dict.items():
            self.log(k, v, prog_bar=True)
        loss = sum(loss_dict.values())
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true, symm_info = batch
        y_true = Symmetrize.apply(y_true, symm_info[1].cpu(), symm_info[2].cpu())
        loss_dict = {}
        loss_dict.update(self.generator_step(x, y_true, symm_info))
        loss_dict.update(self.discriminator_step(x, y_true, symm_info))
        for k, v in loss_dict.items():
            self.log("val_" + k, v)
        loss = sum(loss_dict.values())
        self.log("val_loss", loss, prog_bar=True)
        mask_true = (y_true[:, -2:-1] ** 2) < EPS
        mask_pred = self.y_pred[:, 0:1] < 0
        y_true_vis = torch.cat([
            (y_true[0, -2:-1] ** 2) > EPS,
            torch.masked_fill(y_true, mask_true, 0)[0]
        ], 0)
        y_pred_vis = torch.cat([
            torch.sigmoid(self.y_pred[0, 0:1]),
            torch.masked_fill(self.y_pred[:, 1:], mask_pred, 0)[0]
        ], 0)
        log_predicted_images(self.trainer, x[0, :self.hparams.n_in], y_true_vis, y_pred_vis)
        return loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        b1 = self.hparams.betas[0]
        b2 = self.hparams.betas[1]

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []
