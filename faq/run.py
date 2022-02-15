import os
import json

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    ModelSummary,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from quaterion import Quaterion
from quaterion.dataset.similarity_data_loader import PairsSimilarityDataLoader

from faq.config import DATA_DIR, ROOT_DIR
from faq.datasets.faq_dataset import FAQDataset
from faq.utils.utils import worker_init_fn


def run(
    model,
    train_dataset_path,
    val_dataset_path,
    params,
    use_gpu=torch.cuda.is_available(),
):
    serialization_dir = params.get("serialization_dir", "ckpts")
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss", mode="min", verbose=True, dirpath=serialization_dir,
    )
    sources = {"aws", "azure", "ibm", "yandex_cloud", "hetzner", "gcp"}

    prefix = "full"
    for source in sources:
        if source in os.path.basename(train_dataset_path):
            prefix = source
            break
    logger = params.get("logger")
    if logger == "wandb":
        import uuid

        hparams = {key: params.get(key) for key in ("batch_size", "lr")}
        model_name = f"{model.__class__.__name__}_{prefix}_{str(uuid.uuid4())[:8]}"
        save_dir = os.path.join(ROOT_DIR, "wandb_results")
        os.makedirs(save_dir, exist_ok=True)

        logger = WandbLogger(
            name=model_name,
            project="faq-loss-comparison",
            config=hparams,
            save_dir=save_dir,
        )
    elif logger == "tensorboard":
        logger = TensorBoardLogger(
            os.path.join(serialization_dir, "logs"), name="gated"
        )
    else:
        logger = None

    trainer = pl.Trainer(
        # enable_checkpointing=False,
        callbacks=[
            # checkpoint_callback,
            ModelSummary(max_depth=3),
            EarlyStopping("validation_loss", patience=7),
        ],
        min_epochs=params.get("min_epochs", 1),
        max_epochs=params.get("max_epochs", 150),
        auto_select_gpus=use_gpu,
        log_every_n_steps=params.get("log_every_n_steps", 1),
        gpus=int(use_gpu),
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
    wrong_answers(
        trainer,
        model,
        train_loader,
        valid_loader,
        train_dataset_path,
        val_dataset_path,
    )

    model_class = model.__class__.__name__
    os.makedirs(os.path.join(DATA_DIR, "results", model_class), exist_ok=True)
    with open(
        os.path.join(DATA_DIR, "results", model_class, f"{prefix}.jsonl"), "w"
    ) as f:
        json.dump(
            {key: value.item() for key, value in model.metric.compute().items()},
            f,
            indent=2,
        )
    import wandb

    wandb.finish()


def wrong_answers(
    trainer,
    model,
    train_loader=None,
    valid_loader=None,
    train_filename="",
    val_filename="",
):
    def map_wrong_answers(sentences_filename, indices_filename, res_filename):
        import json

        with open(sentences_filename, "r") as f:
            anchors = []
            others = []

            for j_line in f:
                line = json.loads(j_line)
                anchors.append(line["question"])
                others.append(line["answer"])
        sentences = anchors + others

        with open(indices_filename, "r") as f:
            with open(res_filename, "w") as w:
                for j_line in f:
                    line = json.loads(j_line)
                    anchor = int(line["anchor"])
                    wrong = int(line["wrong"])
                    right = int(line["right"])
                    mapped_line = {
                        "anchor": sentences[anchor],
                        "wrong": sentences[wrong],
                        "right": sentences[right],
                    }
                    json.dump(mapped_line, w)
                    w.write("\n")

    model_class = model.__class__.__name__
    os.makedirs(os.path.join(DATA_DIR, "wrong", model_class), exist_ok=True)

    dataloaders = []
    if train_loader is not None:
        dataloaders.append(train_loader)
    if valid_loader is not None:
        dataloaders.append(valid_loader)
    if not dataloaders:
        raise Exception("pass at least 1 dataloader")
    trainer.predict(model, dataloaders)

    sources = {"aws", "azure", "ibm", "yandex_cloud", "hetzner", "gcp"}

    if train_loader is not None:

        filename = os.path.basename(train_filename)
        prefix = "full"

        for source in sources:
            if source in filename:
                prefix = source
                break
        map_wrong_answers(
            train_filename,
            "train_wrong_predictions.jsonl",  # current dir, created in predict_step
            os.path.join(
                DATA_DIR, "wrong", model_class, f"{prefix}_train_wrong_sentences.jsonl"
            ),
        )
        os.remove("train_wrong_predictions.jsonl")
    if valid_loader is not None:
        filename = os.path.basename(val_filename)
        prefix = "full"
        for source in sources:
            if source in filename:
                prefix = source
                break
        map_wrong_answers(
            val_filename,
            "valid_wrong_predictions.jsonl",  # current dir, created in predict_step
            os.path.join(
                DATA_DIR, "wrong", model_class, f"{prefix}_val_wrong_sentences.jsonl"
            ),
        )
        os.remove("valid_wrong_predictions.jsonl")


if __name__ == "__main__":
    from pytorch_lightning import seed_everything

    seed_everything(42, workers=True)
    pretrained_name = "all-MiniLM-L6-v2"

    parameters = {
        "min_epochs": 1,
        "max_epochs": 150,
        "serialization_dir": "ckpts",
        "lr": 0.01,
        "logger": "wandb",
        # "batch_size": 1000,
    }


    from faq.models.gated import GatedModel
    from faq.models.projector import ProjectorModel
    from faq.models.stacked_model import StackedModel
    from faq.models.skip_connection import SkipConnectionModel

    import time

    by_source_path = os.path.join(DATA_DIR, "by_source")
    # paths = (
    #     (
    #         os.path.join(by_source_path, f"{prefix}_train_cloud_faq_dataset.jsonl"),
    #         os.path.join(by_source_path, f"{prefix}_val_cloud_faq_dataset.jsonl"),
    #     )
    #     for prefix in (
    #     "yandex_cloud",
    #     "hetzner",
    #     "gcp",
    #     "azure",
    #     "ibm",
    #     "aws"
    # )
    # )
    paths = (
        (
            os.path.join(DATA_DIR, "train_cloud_faq_dataset.jsonl"),
            os.path.join(DATA_DIR, "val_cloud_faq_dataset.jsonl")
        ),
    )
    for pair in paths:
        train_path, val_path = pair
        for model_class in (
            GatedModel,
            ProjectorModel,
            StackedModel,
            SkipConnectionModel,
        ):
            model_ = model_class(
                pretrained_name=pretrained_name, lr=parameters.get("lr", 10e-2)
            )

            a = time.perf_counter()
            print("pipeline instantiated")

            run(model_, train_path, val_path, parameters)
            print(time.perf_counter() - a)
            print("pipeline finished")
            time.sleep(2)
