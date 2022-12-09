import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from marinedebrisdetector.transforms import get_transform

from .floatingobjects import FloatingSeaObjectDataset
from .s2ships import S2Ships
from .refined_floatingobjects import RefinedFlobsDataset, RefinedFlobsQualitativeDataset
from .plastic_litter_project import PLPDataset
from .marida import MaridaDataset

class MarineDebrisDataModule(pl.LightningDataModule):
    def __init__(self, data_root: str = "/data/marinedebris",
                 batch_size: int = 32,
                 augmentation_intensity: int = 1,
                 image_size: int = 64,
                 workers: int = 16,
                 no_label_refinement=False,
                 no_s2ships=False,
                 no_marida=False,
                 hr_only=False):

        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.augmentation_intensity = augmentation_intensity
        self.image_size = image_size
        self.image_load_size = int(self.image_size * 1.2)
        self.workers = workers
        self.hr_only = hr_only

        #label-refinement
        self.no_label_refinement = no_label_refinement
        self.no_s2ships = no_s2ships
        self.no_marida = no_marida

        self.flobs_path = os.path.join(self.data_root, "floatingobjects")
        self.refined_flobs_path = os.path.join(self.data_root, "marinedebris_refined")
        self.s2ships_path = os.path.join(self.data_root, "S2SHIPS")
        self.plp_path = os.path.join(self.data_root, "PLP")
        self.maridapath = os.path.join(self.data_root, "MARIDA")

        # scenes to be predicted
        self.durban_scene = os.path.join(self.data_root, "durban", "durban_20190424.tif")
        self.durban_l1c_scene = os.path.join(self.data_root, "durban", "durban_20190424_l1c.tif")
        self.accra_scene = os.path.join(self.data_root, "marinedebris_refined", "accra_20181031.tif")

    def setup(self, stage=None):
        train_transform = get_transform("train", intensity=self.augmentation_intensity, cropsize=self.image_size)
        image_load_size = int(self.image_size * 1.2)  # load images slightly larger to be cropped later to image_size
        flobs_dataset = FloatingSeaObjectDataset(self.flobs_path, fold="train",
                                                 transform=train_transform, refine_labels=not self.no_label_refinement,
                                                 output_size=image_load_size)
        shipsdataset = S2Ships(self.s2ships_path, imagesize=image_load_size, transform=train_transform)
        maridadataset = MaridaDataset(self.maridapath, imagesize = image_load_size, data_transform=train_transform, fold="train")

        train_datasets = [flobs_dataset]
        if not self.no_s2ships:
            train_datasets += [shipsdataset]
        if not self.no_marida:
            train_datasets += [maridadataset]

        test_transform = get_transform("test", cropsize=self.image_size)
        self.train_dataset = ConcatDataset(train_datasets)
        self.valid_dataset = ConcatDataset([
            RefinedFlobsDataset(root=self.refined_flobs_path, fold="val", shuffle=True),
            MaridaDataset(self.maridapath, fold="val",
                          imagesize=self.image_size,
                          data_transform=test_transform,
                          classification=True)
        ])
        maridatestdataset = MaridaDataset(self.maridapath, fold="test",
                                          imagesize = self.image_size,
                                          data_transform=test_transform,
                                          classification=True)
        flobstestdataset = RefinedFlobsDataset(root=self.refined_flobs_path,
                                               fold="test", shuffle=True)
        self.test_dataset = ConcatDataset([flobstestdataset, maridatestdataset])

    def get_qualitative_validation_dataset(self, output_size=256):
        return RefinedFlobsQualitativeDataset(root=self.refined_flobs_path, fold="val", output_size=output_size)

    def get_qualitative_test_dataset(self, output_size=256):
        return RefinedFlobsQualitativeDataset(root=self.refined_flobs_path, fold="test", output_size=output_size)

    def get_prediction_scene_paths(self):
        return [
                    ("durban", self.durban_scene),
                    ("durban_l1c", self.durban_l1c_scene),
                    ("accra", self.accra_scene)
                ]

    def get_plp_dataset(self, year, output_size=32):
        return PLPDataset(root=self.plp_path, year=year, output_size=output_size)


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.workers,
                          shuffle=False,
                          drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.workers,
                          shuffle=False,
                          drop_last=False)
