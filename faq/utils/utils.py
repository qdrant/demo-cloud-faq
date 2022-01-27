from torch.utils.data import get_worker_info


def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    dataset.worker_id = worker_id
    dataset.total_workers = worker_info.num_workers

    dataset_size = worker_info.dataset.size
    batch_size = dataset_size // worker_info.num_workers
    dataset.start_index = batch_size * worker_id
    dataset.end_index = dataset.start_index + batch_size
    if dataset.worker_id == dataset.total_workers - 1:
        dataset.end_index = dataset_size
