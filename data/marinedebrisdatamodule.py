import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from transforms import get_transform
from data.floatingobjects import FloatingSeaObjectDataset
from data.s2ships import S2Ships
from data.refined_floatingobjects import RefinedFlobsDataset, RefinedFlobsQualitativeDataset
from data.plastic_litter_project import PLPDataset

class MarineDebrisDataModule(pl.LightningDataModule):
    def __init__(self, data_root: str = "/data/marinedebris",
                 batch_size: int = 32,
                 augmentation_intensity: int = 1,
                 image_size: int = 64,
                 workers: int = 16):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.augmentation_intensity = augmentation_intensity
        self.image_size = image_size
        self.image_load_size = int(self.image_size * 1.2)
        self.workers = workers

        self.flobs_path = os.path.join(self.data_root, "floatingobjects")
        self.refined_flobs_path = os.path.join(self.data_root, "marinedebris_refined")
        self.s2ships_path = os.path.join(self.data_root, "S2SHIPS")
        self.plp_path = os.path.join(self.data_root, "PLP")

    def setup(self, stage: str):
        train_transform = get_transform("train", intensity=self.augmentation_intensity, cropsize=self.image_size)
        image_load_size = int(self.image_size * 1.2)  # load images slightly larger to be cropped later to image_size
        flobs_dataset = FloatingSeaObjectDataset(self.flobs_path, fold="train",
                                                 transform=train_transform, refine_labels=True,
                                                 output_size=image_load_size, cache_to_npy=True)
        shipsdataset = S2Ships(self.s2ships_path, imagesize=image_load_size, transform=train_transform)
        self.train_dataset = ConcatDataset([flobs_dataset, shipsdataset])
        self.valid_dataset = RefinedFlobsDataset(root=self.refined_flobs_path, fold="val")
        self.test_dataset = RefinedFlobsDataset(root=self.refined_flobs_path, fold="test")

    def get_qualitative_validation_dataset(self):
        return RefinedFlobsQualitativeDataset(root=self.refined_flobs_path, fold="val")

    def get_qualitative_test_dataset(self):
        return RefinedFlobsQualitativeDataset(root=self.refined_flobs_path, fold="test")

    def get_plp_dataset(self, year):
        return PLPDataset(root=self.plp_path, year=year, output_size=32)


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.workers,
                          shuffle=True,
                          drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.workers,
                          shuffle=True,
                          drop_last=False)