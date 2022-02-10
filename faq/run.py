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

from faq.config import DATA_DIR
from faq.datasets.faq_dataset import FAQDataset

from faq.utils.utils import worker_init_fn


def run(model, train_dataset_path, val_dataset_path, params):
    serialization_dir = params.get("serialization_dir", "ckpts")
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss",
        mode="min",
        verbose=True,
        dirpath=serialization_dir,
    )

    logger = params.get("logger")
    if logger == "wandb":
        logger = WandbLogger(project="faq")
    else:
        logger = TensorBoardLogger(
            os.path.join(serialization_dir, "logs"), name="gated"
        )
    trainer = pl.Trainer(
        callbacks=[
            checkpoint_callback,
            ModelSummary(max_depth=3),
            EarlyStopping("validation_loss"),
        ],
        min_epochs=params.get("min_epochs", 1),
        max_epochs=params.get("max_epochs", 50),
        auto_select_gpus=True,
        log_every_n_steps=params.get("log_every_n_steps", 1),
        gpus=1,
        fast_dev_run=False,
        logger=logger,
    )
    train_samples_dataset = FAQDataset(train_dataset_path,)
    valid_samples_dataset = FAQDataset(val_dataset_path,)

    train_loader = PairsSimilarityDataLoader(
        train_samples_dataset,
        batch_size=params.get("batch_size", train_samples_dataset.size),
        worker_init_fn=worker_init_fn,
    )
    valid_loader = PairsSimilarityDataLoader(
        valid_samples_dataset,
        batch_size=params.get("batch_size", valid_samples_dataset.size),
        worker_init_fn=worker_init_fn,
    )
    Quaterion.fit(model, trainer, train_loader, valid_loader)


if __name__ == "__main__":
    from pytorch_lightning import seed_everything

    seed_everything(42, workers=True)
    pretrained_name = "all-MiniLM-L6-v2"

    parameters = {
        "min_epochs": 2,
        "max_epochs": 100,
        "serialization_dir": "ckpts",
        "lr": 0.01,
        "logger": "wandb",
        "batch_size": 1000,
    }
    train_path = os.path.join(
        DATA_DIR, "train_cloud_faq_dataset.jsonl"
    )
    val_path = os.path.join(DATA_DIR, "val_cloud_faq_dataset.jsonl")

    from faq.models.gated import GatedModel

    model_ = GatedModel(
        pretrained_name=pretrained_name, lr=parameters.get("lr", 10e-2),
    )
    # from faq.models.stacked_model import StackedModel
    # model = StackedModel(
    #         pretrained_name=pretrained_name, lr=params.get("lr", 10e-2),
    #     )
    # from faq.models.projector import ProjectorModel
    # model = ProjectorModel(
    #         pretrained_name=pretrained_name, lr=params.get("lr", 10e-2),
    #     )
    #
    # from faq.models.skip_connection import SkipConnectionModel
    # model = SkipConnectionModel(
    #         pretrained_name=pretrained_name, lr=params.get("lr", 10e-2),
    #     )

    import time

    a = time.perf_counter()
    print("pipeline instantiated")

    run(model_, train_path, val_path, parameters)
    print(time.perf_counter() - a)
    print("pipeline finished")
