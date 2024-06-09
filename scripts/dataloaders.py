from datasets import load_dataset
import json

def load_huggingface_dataset(dataset_path, subset_name, split_name, num_head):
    """ Load a dataset from the Hugging Face datasets library. """
    if subset_name:
        dataset = load_dataset(dataset_path, subset_name, split=split_name)
    else:
        dataset = load_dataset(dataset_path, split=split_name)
    dataset = [{k: v for k, v in item.items()} for item in dataset][:num_head]
    return dataset

def load_custom_dataset(dataset_path, num_head):
    """ Load a custom dataset from a JSONL file. """
    if dataset_path.endswith(".jsonl"):
        with open(dataset_path, "r") as f:
            dataset = [json.loads(line) for line in f][:num_head]
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}")
    return dataset


def load_evaldataset(config):
    """ Load a dataset for evaluation. """
    dataset_type = config.get('dataset_type', None)
    dataset_path = config.get('dataset_path', None)
    if dataset_path is None:
        raise ValueError("dataset_path must be specified in the config file.")
    subset_name = config.get('subset_name', None)
    split_name = config.get('split_name', 'test')
    num_head = config.get('num_head', None)

    # dataset type autodetection
    if dataset_type is None:
        if dataset_path.endswith(".jsonl"):
            dataset_type = "custom"
        else:
            dataset_type = "huggingface"

    # main
    if dataset_type == "huggingface":
        return load_huggingface_dataset(dataset_path, subset_name, split_name, num_head)
    elif dataset_type == "custom":
        return load_custom_dataset(dataset_path, num_head)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
