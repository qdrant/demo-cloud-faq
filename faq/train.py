import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    ModelSummary,
)

from quaterion import Quaterion
from quaterion.dataset import PairsSimilarityDataLoader

from faq.dataset import FAQDataset


def run(model, train_dataset_path, val_dataset_path, params):
    checkpoint_dir = params.get("checkpoint_dir", "checkpoints")
    use_gpu = params.get("cuda", torch.cuda.is_available())

    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss", mode="min", verbose=True, dirpath=checkpoint_dir,
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, ModelSummary(max_depth=3),],
        min_epochs=params.get("min_epochs", 1),
        max_epochs=params.get("max_epochs", 150),
        auto_select_gpus=use_gpu,
        log_every_n_steps=params.get("log_every_n_steps", 1),
        gpus=int(use_gpu),
    )

    train_dataset = FAQDataset(train_dataset_path)
    val_dataset = FAQDataset(val_dataset_path)

    train_dataloader = PairsSimilarityDataLoader(
        train_dataset, batch_size=params.get("train_batch_size", len(train_dataset))
    )
    val_dataloader = PairsSimilarityDataLoader(
        val_dataset, batch_size=params.get("val_batch_size", len(val_dataset))
    )

    Quaterion.fit(model, trainer, train_dataloader, val_dataloader)


if __name__ == "__main__":
    import os

    from pytorch_lightning import seed_everything

    from faq.model import FAQModel
    from faq.config import DATA_DIR, ROOT_DIR

    seed_everything(42, workers=True)

    pretrained_name = "all-MiniLM-L6-v2"
    learning_rate = 10e-2
    parameters = {
        "max_epochs": 2,
        "train_batch_size": 16,
        "val_batch_size": 16,
        "checkpoint_dir": os.path.join(ROOT_DIR, "checkpoints"),
    }
    faq_model = FAQModel(pretrained_name=pretrained_name, lr=learning_rate)
    train_path = os.path.join(DATA_DIR, "train_cloud_faq_dataset.jsonl")
    val_path = os.path.join(DATA_DIR, "val_cloud_faq_dataset.jsonl")
    run(faq_model, train_path, val_path, parameters)
    faq_model.save_servable(os.path.join(ROOT_DIR, "servable"))
