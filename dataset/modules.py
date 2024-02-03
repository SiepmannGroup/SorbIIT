import h5py
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .factory import DatasetFactory
from .datasets import SpatialDerivativeDataset as Dataset

class ZTBDatasetModule(pl.LightningDataModule):
    
    def __init__(
        self,
        train_size,
        val_size,
        molecule,
        pool_level=2,
        solver="quadratic",
        num_workers=4,
    ):
        super().__init__()
        f = DatasetFactory(molecule)
        self.keys = []
        self.pool_level = pool_level
        self.solver = solver
        with h5py.File(f.mol_config["simulation_file"], 'r') as h5file:
            # skip unit cells that are too large
            for key in h5file.keys():
                if key[:-6] not in self.keys:
                    shape = np.array(
                        h5file[key].get('grid_sizes') or h5file[key].attrs.get('grid_sizes')
                    )
                    if np.prod(shape) < 2e7:
                        self.keys.append(key[:-6])                    
        self.train_size = train_size
        self.val_size = val_size
        self.train_keys = self.keys[:self.train_size]
        self.val_keys = self.keys[self.train_size : self.train_size + self.val_size]
        self.num_workers = num_workers

        self.training_set = f.create(self.pool_level, self.train_keys, self.solver)
        self.validation_set = f.create(self.pool_level, self.val_keys, self.solver)
        
        self.atoms = self.training_set.atoms

    def get_shapes(self):
        x, y, _ = self.training_set[0]
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
