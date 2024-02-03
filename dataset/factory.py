import yaml
import pathlib

from .datasets import SpatialDerivativeDataset
from utils import TraPPE

config_path = pathlib.Path(__file__).parent.joinpath("dataset_config.yaml")

class DatasetFactory:

    def __init__(self, molecule):
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.mol_config = self.config["molecules"][molecule]
    
    def create(self, pool_level=2, keys=None, solver="quadratic"):
        molecules = [getattr(TraPPE, x) for x in self.mol_config["atoms"]]
        ds = SpatialDerivativeDataset(
                molecules,
                pool_level,
                (64, 64),  # fixed input/output size, deprecated
                self.mol_config["simulation_file"],
                self.config["ztb_path"],
                self.mol_config["pressures"],
                self.mol_config["temperatures"],
                keys=keys,
                ntrans=0,
                nrot=0,
                axisrot=True,
                cif_path=self.config["cif_path"],
                positional_encoding=None,
                normalize_all=False,
                return_symmetrize_func=True,
                symmetrize_in_place=False,
                solver=solver
            )
        return ds


def create_dataset(molecule, pool_level=2, keys=None, solver="quadratic"):
    return DatasetFactory(molecule).create(pool_level, keys, solver)

