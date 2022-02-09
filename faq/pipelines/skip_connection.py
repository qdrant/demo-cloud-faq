import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    ModelSummary,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from quaterion import Quaterion
from quaterion.dataset.similarity_data_loader import PairsSimilarityDataLoader

from faq.datasets.faq_dataset import FAQDataset
from faq.models.skip_connection import SkipConnectionModel
from faq.utils.utils import worker_init_fn


class SkipConnectionModelPipeline:
    def __init__(
        self,
        train_dataset_path,
        val_dataset_path,
        params,
        pretrained_name="all-MiniLM-L6-v2",
    ):
        self.model = SkipConnectionModel(
            pretrained_name=pretrained_name, lr=params.get("lr", 10e-2),
        )
        self.serialization_dir = params.get("serialization_dir", "ckpts")
        self.params = params
        self._train_dataset_path = train_dataset_path
        self._val_dataset_path = val_dataset_path

    def run(self):

        checkpoint_callback = ModelCheckpoint(
            monitor="validation_loss",
            mode="min",
            verbose=True,
            dirpath=self.serialization_dir,
        )

        logger = self.params.get("logger")
        if logger == "wandb":
            logger = WandbLogger(project="faq")
        else:
            logger = TensorBoardLogger(
                os.path.join(self.serialization_dir, "logs"), name="gated"
            )
        trainer = pl.Trainer(
            callbacks=[
                checkpoint_callback,
                ModelSummary(max_depth=3),
                EarlyStopping("validation_loss"),
            ],
            min_epochs=self.params.get("min_epochs", 1),
            max_epochs=self.params.get("max_epochs", 50),
            auto_select_gpus=True,
            log_every_n_steps=self.params.get("log_every_n_steps", 1),
            gpus=1,
            fast_dev_run=False,
            logger=logger,
        )
        train_samples_dataset = FAQDataset(self._train_dataset_path,)
        valid_samples_dataset = FAQDataset(self._val_dataset_path,)

        train_loader = PairsSimilarityDataLoader(
            train_samples_dataset,
            batch_size=self.params.get(
                "batch_size", train_samples_dataset.size
            ),
            worker_init_fn=worker_init_fn,
        )
        valid_loader = PairsSimilarityDataLoader(
            valid_samples_dataset,
            batch_size=self.params.get(
                "batch_size", valid_samples_dataset.size
            ),
            worker_init_fn=worker_init_fn,
        )
        Quaterion.fit(self.model, trainer, train_loader, valid_loader)
