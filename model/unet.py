import taichi as ti
import torch
from torch import nn
from .kernels import symmetrize_kernel, pad_kernel


class Symmetrize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, cell_vectors, transforms):
        y = torch.zeros_like(x, device=x.device)
        symmetrize_kernel(x, y, cell_vectors.to(x.device), transforms.to(x.device))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        Note that symmetrization can be viewed as a linear operator.
        Therefore, the backward pass of symmetrization is also the same
        operation. Since the input adjoint is already symmetrized (if
        the model is trained against perfectly symmetric outputs),
        the backward pass can simply do nothing.
        """
        return grad_output, None, None

class UnitCellPad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, cell_vectors, pad_amount):
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
        pad_kernel(y, cell_vectors.to(x.device), *pad_amount)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class UnitCellConv3d(nn.Module):

    def __init__(self, *args, symmetrize=False, **kwargs):
        super().__init__()
        if "pad" in kwargs:
            self.pad = kwargs.pop("pad")
        else:
            self.pad = 0
        self.conv = nn.Conv3d(*args, **kwargs)
        self.symmetrize = symmetrize

    def forward(self, x, cell_vectors, transforms):
        # pad
        if self.pad > 0:
            y = UnitCellPad.apply(x, cell_vectors, self.pad)
        else:
            y = x
        # conv
        y = self.conv(y)
        # symmetrize
        if self.symmetrize:
            y = Symmetrize.apply(y, cell_vectors, transforms)
        return y

class UnitCellDoubleConv(nn.Module):

    def __init__(self, c_in, c_mid, c_out, padding):
        super().__init__()
        self.conv1 = UnitCellConv3d(c_in, c_mid, 3, stride=1, padding=padding)
        self.bn1 = nn.BatchNorm3d(c_mid, affine=False)
        self.conv2 = UnitCellConv3d(c_mid, c_out, 3, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm3d(c_out, affine=False)
        self.relu = nn.ReLU()

    def forward(self, x, cell_vectors, transforms):
        y = self.relu(self.bn1(self.conv1(x, cell_vectors, transforms)))
        y = self.relu(self.bn2(self.conv2(y, cell_vectors, transforms)))
        return y

class CellUNet(nn.Module):

    '''
    original paper: channels=[in, 32, 64, 128, 256, 512, out]
    '''
    def __init__(self, n_in, n_out, channels=(16, 32, 64, 128)):
        super().__init__()
        self.channels = [n_in, *channels, n_out]
        self.nblocks = len(channels) - 1
        if self.nblocks <= 0:
            raise ValueError("not enough layers!")
        
        self.down = nn.ModuleList([])
        self.up = nn.ModuleList([])
        self.conv_tr = nn.ModuleList([])
        self.relu = nn.ReLU()

        self.inp = UnitCellDoubleConv(*self.channels[0:3], padding=1)
    
        for i in range(1, self.nblocks):
            self.down.append(UnitCellDoubleConv(
                self.channels[i + 1],
                self.channels[i + 1],
                self.channels[i + 2],
                padding=1,
            ))
            self.up.append(UnitCellDoubleConv(
                self.channels[i + 1] + self.channels[i + 2],
                self.channels[i + 1],
                self.channels[i + 1],
                padding=1,
            ))
            self.conv_tr.append(
                UnitCellConv3d(self.channels[i + 2], self.channels[i + 2], 1, stride=1, padding=0),
            )
            
        self.pool = nn.MaxPool3d(2, 2, padding=0)
        self.out = UnitCellConv3d(self.channels[2],
            self.channels[-1], 1, stride=1, padding=0, symmetrize=True)

    def forward(self, x, cell_vectors, transforms, return_latent=False):
        z = self.inp(x, cell_vectors, transforms)
        cell_new = cell_vectors
        zs = []
        for i in range(self.nblocks - 1):
            zs.append(z)
            z = self.pool(z)
            cell_new = cell_new * (torch.tensor(z.shape[2:]) / torch.tensor(zs[-1].shape[2:])).unsqueeze(1)
            z = self.down[i](z, cell_new, transforms)
        y = z
        for i in range(1, self.nblocks):
            old_shape = y.shape[2:]
            # upsample to the exact size of corresponding input layer
            y = torch.nn.functional.interpolate(
                y, zs[-i].shape[2:], mode='trilinear', align_corners=False
            )
            cell_new = cell_new * (torch.tensor(y.shape[2:]) / torch.tensor(old_shape)).unsqueeze(1)
            y = self.relu(self.conv_tr[-i](y, cell_new, transforms))
            y = torch.cat([zs[-i], y], dim=1)
            y = self.up[-i](y, cell_new, transforms)
        y = self.out(y, cell_new, transforms)
        return (y, z) if return_latent else y
    
    def feature(self, x):
        z = self.inp(x)
        for i in range(self.nblocks - 1):
            z = self.pool(z)
            z = self.down[i](z)
        return z
