import torch

class PatchConv3D(torch.nn.Module):

    def __init__(
        self,
        channel_in,
        hidden_layers=3,
        kernel_size=4,
        pad_size=1,
        n_filters=16,
    ):
        super().__init__()

        sequence = [torch.nn.Conv3d(channel_in, n_filters, kernel_size, stride=2, padding=pad_size),
                    #torch.nn.BatchNorm3d(ndf, affine=False),
                    torch.nn.LeakyReLU(0.2, True)]
        for n in range(hidden_layers):
            sequence += [torch.nn.Conv3d(n_filters, n_filters * 2, kernel_size, stride=2, padding=pad_size),
                        #torch.nn.BatchNorm3d(ndf * 2, affine=False),
                        torch.nn.LeakyReLU(0.2, True)]
            n_filters *= 2
        sequence += [torch.nn.Conv3d(n_filters, n_filters, kernel_size, stride=1, padding=pad_size),
                        torch.nn.BatchNorm3d(n_filters, affine=False),
                        #torch.nn.LeakyReLU(0.2, True),
                        torch.nn.Conv3d(n_filters, 1, kernel_size, stride=1, padding=pad_size)]
        self.model = torch.nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)