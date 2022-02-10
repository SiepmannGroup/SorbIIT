import h5py
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .datasets import SpatialDerivativeDataset as Dataset

class ZTBDatasetModule(pl.LightningDataModule):
    
    def __init__(
        self,
        train_size,
        val_size,
        atoms,
        pool_level,
        window_size,
        traj_path,
        ztb_path,
        pressures,
        temperatures,
        ntrans,
        nrot=0,
        axisrot=True,
        cif_path=None,
        positional_encoding="base",
        normalize_all=True,
        return_symmetrize_func=False,
        num_workers=4,
        symmetrize_in_place=True,
    ):
        super().__init__()
        self.keys = []
        with h5py.File(traj_path, 'r') as h5file:
            # skip unit cells that are too large
            for key in h5file.keys():
                if key[:-6] not in self.keys:
                    shape = np.array(h5file[key]['grid_sizes'])
                    if np.prod(shape) < 2e7:
                        self.keys.append(key[:-6])                    
    
        self.train_size = train_size
        self.val_size = val_size
        self.train_keys = self.keys[:self.train_size]
        self.val_keys = self.keys[self.train_size : self.train_size + self.val_size]
        self.num_workers = num_workers
        self.symmetrize_in_place = symmetrize_in_place

        if isinstance(window_size, int):
            window_size = (window_size, window_size)

        self.training_set = Dataset(
            atoms,
            pool_level,
            window_size,
            traj_path,
            ztb_path,
            pressures,
            temperatures,
            keys=self.train_keys,
            ntrans=ntrans,
            nrot=nrot,
            axisrot=axisrot,
            cif_path=cif_path,
            positional_encoding=positional_encoding,
            normalize_all=normalize_all,
            return_symmetrize_func=return_symmetrize_func,
            symmetrize_in_place=symmetrize_in_place,
        )

        self.validation_set = Dataset(
            atoms,
            pool_level,
            window_size,
            traj_path,
            ztb_path,
            pressures,
            temperatures,
            keys=self.val_keys,
            ntrans=ntrans,
            nrot=nrot,
            axisrot=axisrot,
            cif_path=cif_path,
            positional_encoding=positional_encoding,
            normalize_all=normalize_all,
            return_symmetrize_func=return_symmetrize_func,
            symmetrize_in_place=symmetrize_in_place,
        )

    def get_shapes(self):
        x, y = self.training_set.sample(1, 0, 0, 0)
        return x.shape, y.shape

    def train_dataloader(self):
        return DataLoader(
            self.training_set, 
            batch_size=1,
            pin_memory=True,
            collate_fn=lambda x: x[0],
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_set,
            batch_size=1,
            pin_memory=True,
            collate_fn=lambda x: x[0],
            num_workers=self.num_workers
        )
