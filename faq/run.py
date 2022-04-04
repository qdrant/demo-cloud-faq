import os
import glob
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
    test_dataset_path,
    params,
    use_gpu=torch.cuda.is_available(),
    loss_fn="mnr",
):
    model_class = model.__class__.__name__
    serialization_dir = os.path.join(
        params.get("serialization_dir", "ckpts"), loss_fn, model_class
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss",
        mode="min",
        verbose=True,
        dirpath=serialization_dir,
        filename="epoch{epoch:02d}-val_loss{validation_loss:.4f}",
        auto_insert_metric_name=False,
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
        model_name = f"{model_class}_{prefix}_{str(uuid.uuid4())[:8]}"
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
            os.path.join(serialization_dir, "logs"), name=f"{model_class}_{loss_fn}"
        )
    else:
        logger = None

    trainer = pl.Trainer(
        # enable_checkpointing=False,
        callbacks=[
            checkpoint_callback,
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
    test_samples_dataset = FAQDataset(test_dataset_path,)

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

    if not params["testing"]:  # simply train and return
        Quaterion.fit(model, trainer, train_loader, valid_loader)

    test_loader = PairsSimilarityDataLoader(
        test_samples_dataset,
        batch_size=params.get("batch_size", test_samples_dataset.size),
        worker_init_fn=worker_init_fn
    )

    wrong_answers(
        trainer,
        model,
        test_loader,
        test_dataset_path,
        serialization_dir,
    )

    os.makedirs(os.path.join(serialization_dir, "results"), exist_ok=True)
    with open(os.path.join(serialization_dir, "results", f"{prefix}.jsonl"), "w") as f:
        json.dump(
            {stage: {key: value.item() for key, value in metric.items()} for stage, metric in model.metric_last_state.items()},
            f,
            indent=2,
        )

    if logger == "wandb":
        import wandb

        wandb.finish()


def wrong_answers(
    trainer,
    model,
    test_loader,
    test_filename,
    serialization_dir="",
):
    checkpoint_path = glob.glob(f"{serialization_dir}/*.ckpt")[0]
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    model.setup_dataloader(test_loader)

    model.cache(
        trainer=trainer, train_dataloader=test_loader, val_dataloader=None
    )

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

    os.makedirs(os.path.join(serialization_dir, "wrong"), exist_ok=True)

    trainer.test(model, [test_loader])

    sources = {"aws", "azure", "ibm", "yandex_cloud", "hetzner", "gcp"}

    filename = os.path.basename(test_filename)
    prefix = "full"

    for source in sources:
        if source in filename:
            prefix = source
            break

    map_wrong_answers(
        test_filename,
        f"wrong_predictions.jsonl",  # current dir, created in test_step
        os.path.join(
            serialization_dir, "wrong", f"{prefix}_wrong_sentences.jsonl"
        ),
    )
    os.remove(f"wrong_predictions.jsonl")


if __name__ == "__main__":
    from pytorch_lightning import seed_everything

    seed_everything(42, workers=True)
    pretrained_name = "all-MiniLM-L6-v2"

    from faq.models.gated import GatedModel
    from faq.models.widening import WideningModel
    from faq.models.projector import ProjectorModel
    from faq.models.stacked_model import StackedModel
    from faq.models.skip_connection import SkipConnectionModel

    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--testing", action="store_true", default=False, help="Test trained models",
    )
    ap.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    ap.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    ap.add_argument(
        "--min_epochs",
        type=int,
        default=1,
        help="Minimum number of epochs to run training",
    )
    ap.add_argument(
        "--max_epochs",
        type=int,
        default=150,
        help="Maximum number of epochs to run training",
    )
    ap.add_argument(
        "--serialization_dir",
        default="ckpts",
        help="Directory to save checkpoints and logs",
    )
    ap.add_argument(
        "--logger", type=str, choices=["tensorboard", "wandb"], default="tensorboard"
    )

    parameters = vars(ap.parse_args())

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
            os.path.join(DATA_DIR, "val_cloud_faq_dataset.jsonl"),
            os.path.join(DATA_DIR, "cloud_faq_dataset.jsonl"),
        ),
    )
    for triplet in paths:
        train_path, val_path, test_path = triplet
        for model_class in (
            GatedModel,
            WideningModel,
            # ProjectorModel,
            # StackedModel,
            # SkipConnectionModel,
        ):
            for loss_fn in ["mnr", "contrastive"]:
                model_ = model_class(
                    pretrained_name=pretrained_name,
                    lr=parameters.get("lr", 10e-2),
                    loss_fn=loss_fn,
                )

                a = time.perf_counter()
                print("pipeline instantiated")

                run(model_, train_path, val_path, test_path, parameters, loss_fn=loss_fn)
                print(time.perf_counter() - a)
                print("pipeline finished")
                time.sleep(2)
