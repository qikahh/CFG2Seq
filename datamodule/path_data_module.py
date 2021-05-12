from os.path import exists, join
from typing import List, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from datamodule import PathDataset, PathSample, PathBatch
from utils.vocabulary import Vocabulary


class PathDataModule(LightningDataModule):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__()
        self._config = config
        self._vocabulary = vocabulary

        self._dataset_dir = join(config.data_folder, config.dataset.name)
        self._train_data_file = join(self._dataset_dir, f"{config.train_holdout}.json")
        self._val_data_file = join(self._dataset_dir, f"{config.val_holdout}.json")
        self._test_data_file = join(self._dataset_dir, f"{config.test_holdout}.json")

    def prepare_data(self):
        if not exists(self._dataset_dir):
            raise ValueError(f"There is no file in passed path ({self._dataset_dir})")
        # TODO: download data from s3 if not exists

    def setup(self, stage: Optional[str] = None):
        # TODO: collect or convert vocabulary if needed
        pass

    @staticmethod
    def collate_wrapper(batch: List[PathSample]) -> PathBatch:
        return PathBatch(batch)

    def _create_dataset(self, data_file: str, random_context: bool) -> Dataset:
        return PathDataset(data_file, self._config, self._vocabulary, random_context)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        dataset = self._create_dataset(self._train_data_file, self._config.hyper_parameters.random_context)
        return DataLoader(
            dataset,
            batch_size=self._config.hyper_parameters.batch_size,
            shuffle=self._config.hyper_parameters.shuffle_data,
            num_workers=self._config.num_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        dataset = self._create_dataset(self._val_data_file, False)
        return DataLoader(
            dataset,
            batch_size=self._config.hyper_parameters.test_batch_size,
            shuffle=False,
            num_workers=self._config.num_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        dataset = self._create_dataset(self._test_data_file, False)
        return DataLoader(
            dataset,
            batch_size=self._config.hyper_parameters.batch_size,
            shuffle=False,
            num_workers=self._config.num_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def transfer_batch_to_device(
        self, batch: PathBatch, device: Optional[torch.device] = None
    ) -> PathBatch:
        if device is not None:
            batch.move_to_device(device)
        return batch
