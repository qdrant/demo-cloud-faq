import torch
import pytorch_lightning as pl

from quaterion import Quaterion
from quaterion.dataset import PairsSimilarityDataLoader
from quaterion.eval.evaluator import Evaluator
from quaterion.eval.pair import RetrievalReciprocalRank, RetrievalPrecision
from quaterion.eval.samplers.pair_sampler import PairSampler

from faq.dataset import FAQDataset


def train(model, train_dataset_path, val_dataset_path, params):
    use_gpu = params.get("cuda", torch.cuda.is_available())

    trainer = pl.Trainer(
        min_epochs=params.get("min_epochs", 1),
        max_epochs=params.get("max_epochs", 2),
        auto_select_gpus=use_gpu,
        log_every_n_steps=params.get("log_every_n_steps", 1),
        gpus=int(use_gpu),
        num_sanity_val_steps=0,
    )
    train_dataset = FAQDataset(train_dataset_path)
    val_dataset = FAQDataset(val_dataset_path)
    train_dataloader = PairsSimilarityDataLoader(train_dataset, batch_size=1024)
    val_dataloader = PairsSimilarityDataLoader(val_dataset, batch_size=1024)
    Quaterion.fit(model, trainer, train_dataloader, val_dataloader)

    metrics = {
        "rrk": RetrievalReciprocalRank(),
        "rp@1": RetrievalPrecision(k=1)
    }
    sampler = PairSampler()
    evaluator = Evaluator(metrics, sampler)
    results = Quaterion.evaluate(evaluator, val_dataset, model.model)
    print(f"results: {results}")


if __name__ == "__main__":
    import os
    from pytorch_lightning import seed_everything
    from faq.model import FAQModel
    from faq.config import DATA_DIR, ROOT_DIR

    seed_everything(42, workers=True)
    faq_model = FAQModel()
    train_path = os.path.join(DATA_DIR, "train_part.jsonl")
    val_path = os.path.join(DATA_DIR, "val_part.jsonl")
    train(faq_model, train_path, val_path, {})
    # faq_model.save_servable(os.path.join(ROOT_DIR, "servable"))
