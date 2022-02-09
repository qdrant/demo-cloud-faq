from pipelines import (
    GatedModelPipeline,
    StackedModelPipeline,
    ProjectorModelPipeline,
    SkipConnectionModelPipeline,
)


if __name__ == "__main__":
    from pytorch_lightning import seed_everything

    seed_everything(42, workers=True)

    params = {
        "min_epochs": 2,
        "max_epochs": 100,
        "serialization_dir": "ckpts",
        "lr": 0.01,
        "logger": "wandb",
    }
    # pipeline = GatedModelPipeline(
    #     train_dataset_path="../data/train_cloud_faq_dataset.jsonl",
    #     val_dataset_path="../data/val_cloud_faq_dataset.jsonl",
    #     # train_dataset_path="../data/btrain_part.jsonl",
    #     # val_dataset_path="../data/bval_part.jsonl",
    #     params=params,
    # )
    # pipeline = ProjectorModelPipeline(
    #     train_dataset_path="../data/train_cloud_faq_dataset.jsonl",
    #     val_dataset_path="../data/val_cloud_faq_dataset.jsonl",
    #     # train_dataset_path="../data/btrain_part.jsonl",
    #     # val_dataset_path="../data/bval_part.jsonl",
    #     params=params,
    # )
    # pipeline = SkipConnectionModelPipeline(
    #     train_dataset_path="../data/train_cloud_faq_dataset.jsonl",
    #     val_dataset_path="../data/val_cloud_faq_dataset.jsonl",
    #     # train_dataset_path="../data/btrain_part.jsonl",
    #     # val_dataset_path="../data/bval_part.jsonl",
    #     params=params,
    # )
    pipeline = StackedModelPipeline(
        train_dataset_path="../data/train_cloud_faq_dataset.jsonl",
        val_dataset_path="../data/val_cloud_faq_dataset.jsonl",
        # train_dataset_path="../data/btrain_part.jsonl",
        # val_dataset_path="../data/bval_part.jsonl",
        params=params,
    )

    import time

    a = time.perf_counter()
    print("pipeline instantiated")
    pipeline.run()
    print(time.perf_counter() - a)
    print("pipeline finished")
