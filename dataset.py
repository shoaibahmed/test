#!/bin/python

import os
import psutil
import random
from itertools import chain
from typing import List, Dict

import torch
import numpy as np
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import PreTrainedTokenizer

from dist_checkpoint_utils import StatefulDistributedSampler


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, num_workers: int = 8,
                   drop_last: bool = False, pin_loader_memory: bool = True,
                   persistent_workers: bool = True, prefetch_factor: int = 5,
                   generator: torch.Generator = None, shuffle: bool = True,
                   use_stateful_sampler: bool = True):
    sampler = None
    if torch.distributed.is_initialized():
        if use_stateful_sampler:
            print("!! Attaching stateful sampler to the DataLoader for distributed training...")
            sampler = StatefulDistributedSampler(dataset, batch_size=batch_size, shuffle=shuffle)
        else:
            print("!! Attaching sampler to the DataLoader for distributed training...")
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # shuffling is done by the sampler
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             sampler=sampler, drop_last=drop_last, pin_memory=pin_loader_memory,
                                             worker_init_fn=seed_worker, persistent_workers=persistent_workers,
                                             prefetch_factor=prefetch_factor, generator=generator)
    return dataloader


class NLPDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name: str, tokenizer: PreTrainedTokenizer, max_length: int,
                 sequence_stride: int, combine_documents: bool, logging_level: int = 0,
                 subsample_size: int = -1, sampler_seed: int = 43, include_other_cols: bool = False,
                 num_proc: int = None, create_test_split: bool = False):
        # Load the original dataset
        assert dataset_name in ["pg19", "cc_news", "wikitext-2", "bookcorpus", "c4", "openwebtext", "slimpajama",
                                "fineweb_edu"]
        print("!! Loading dataset:", dataset_name)
        if sequence_stride is None:
            sequence_stride = max_length
        assert sequence_stride <= max_length, f"{sequence_stride} is not <= {max_length}"
        print(f"Dataset config / max length: {max_length} / sequence stride: {sequence_stride}")

        subsample_dataset = True  # Subsample in this case due to large dataset size
        if dataset_name == "pg19":
            dataset = load_dataset("pg19")
        elif dataset_name == "fineweb_edu":
            dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT")
        elif dataset_name == "c4":
            # Load the en-noblocklist subset of C4 (https://huggingface.co/datasets/c4)
            dataset = load_dataset("c4", "en.noblocklist")
        elif dataset_name == "openwebtext":
            dataset = load_dataset("openwebtext")
        elif dataset_name == "slimpajama":
            dataset = load_dataset("cerebras/SlimPajama-627B")
        elif dataset_name == "cc_news":
            dataset = load_dataset("cc_news")
            subsample_dataset = False  # dataset small enough
        elif dataset_name == "bookcorpus":
            dataset = load_dataset("bookcorpus")
        else:
            assert dataset_name == "wikitext-2"
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
            subsample_dataset = False  # dataset small enough

        # Create a test split in case the original test split is not provided
        if "test" not in dataset:
            if create_test_split:  # Split the dataset into training (90%) and testing (10%)
                print("Creating synthetic test split...")
                assert "train" in dataset, dataset
                d = dataset["train"].train_test_split(test_size=0.1, seed=sampler_seed, shuffle=True)
            else:
                print("Only loading the training set without test set...")
                d = dataset
        else:
            print("Using the official test split...")
            d = dataset

        if subsample_dataset and subsample_size is not None and subsample_size > 0:
            # Define the random number generator based on the random seed
            rng = np.random.default_rng(sampler_seed)

            # Subsample the dataset for OpenWebText as a starting point
            train_examples = subsample_size  # 1M examples from the dataset
            print(f"!! Subsampling train dataset to {train_examples} examples...")
            possible_idx = list(range(len(d["train"])))
            selected_idx = rng.choice(possible_idx, size=(train_examples), replace=False)
            d["train"] = d["train"].select(selected_idx)
            if "test" in d:  # test split exists
                eval_examples = int(0.1 * train_examples)
                if len(d["test"]) > eval_examples:
                    print(f"!! Subsampling test dataset to {eval_examples} examples...")
                    possible_idx = list(range(len(d["test"])))
                    selected_idx = rng.choice(possible_idx, size=(eval_examples), replace=False)
                    d["test"] = d["test"].select(selected_idx)
            print("!! Dataset subsampling completed...")

        filter_dataset = True
        if filter_dataset:
            prev_train_size = len(d["train"])
            d["train"] = d["train"].filter(lambda example: len(example["text"]) > 0)
            print(f"Train dataset filtering / old size: {prev_train_size} / new size: {len(d['train'])}")

            if "test" in d:  # test split exists
                prev_test_size = len(d["test"])
                d["test"] = d["test"].filter(lambda example: len(example["text"]) > 0)
                print(f"Test dataset filtering / old size: {prev_test_size} / new size: {len(d['test'])}")

        if logging_level > 0:
            print("Full dataset:", dataset)
            print(f"Splits / train: {d['train']} / test: {d['test'] if 'test' in d else 'None'}")

        if logging_level > 1:
            for t in d["train"]["text"][:3]:
                print(t)
                print("="*50)

        self.max_length = max_length
        self.sequence_stride = sequence_stride
        self.tokenizer = tokenizer

        truncate_longer_samples = False
        num_proc = psutil.cpu_count() if num_proc is None else num_proc
        print(f"# processes for mapping: {num_proc} / combine documents: {combine_documents}")

        # the encode function will depend on the truncate_longer_samples variable
        encode = self.encode_with_truncation if truncate_longer_samples else self.encode_without_truncation

        # tokenizing the train/test dataset (essential to remove columns as they can result in wrong model keys)
        remove_cols = d["train"].column_names
        print("Removing columns:", remove_cols)
        train_dataset = d["train"].map(encode, remove_columns=remove_cols, batched=True, num_proc=num_proc, desc="Train encoding")
        test_dataset = None
        if "test" in d:  # test split exists
            test_dataset = d["test"].map(encode, remove_columns=remove_cols, batched=True, num_proc=num_proc, desc="Test encoding")

        if truncate_longer_samples:
            columns = ["input_ids", "attention_mask"] if include_other_cols else ["input_ids"]
        else:
            columns = ["input_ids", "attention_mask", "special_tokens_mask"] if include_other_cols else ["input_ids"]
        train_dataset.set_format(type="torch", columns=columns)
        if test_dataset is not None:  # test split exists
            test_dataset.set_format(type="torch", columns=columns)

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
        if not truncate_longer_samples:
            print(f"!! Grouping {'combined ' if combine_documents else ''}documents to size: {self.max_length}")
            grouping_func = self.group_texts if combine_documents else self.group_texts_within_documents
            train_dataset = train_dataset.map(grouping_func, batched=combine_documents, num_proc=num_proc,
                                              desc=f"Grouping texts in chunks of {self.max_length}")
            train_dataset.set_format("torch")

            if test_dataset is not None:  # test split exists
                test_dataset = test_dataset.map(grouping_func, batched=combine_documents, num_proc=num_proc,
                                                desc=f"Grouping texts in chunks of {self.max_length}")
                test_dataset.set_format("torch")

        output_ds_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
        if test_dataset is None:  # test split is missing
            del output_ds_dict["test"]
        self.datasets = output_ds_dict

    @staticmethod
    def is_dataset_processed(dataset_dir) -> bool:
        return os.path.exists(os.path.join(dataset_dir, "processed.log"))

    def save_datasets(self, dataset_dir) -> None:
        if not NLPDataset.is_dataset_processed(dataset_dir):
            print("Saving dataset to disk:", dataset_dir)
            self.datasets.save_to_disk(dataset_dir)
            with open(os.path.join(dataset_dir, "processed.log"), "w") as f:  # write the log as a indicator that dataset processing completed
                f.write("done")

    @staticmethod
    def load_dataset(dataset_dir) -> DatasetDict:
        assert NLPDataset.is_dataset_processed(dataset_dir), f"Dataset not processed! Please run the dataset generator first."
        print("Loading dataset from disk:", dataset_dir)
        datasets = load_from_disk(dataset_dir)
        if "test" not in datasets:
            datasets["test"] = None
        return datasets

    def group_texts(self, examples: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Main data processing function that will concatenate all texts from our dataset and generate chunks of max_seq_length.
        # grabbed from: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.max_length:
            total_length = ((total_length - self.max_length) // self.sequence_stride) * self.sequence_stride
            # total_length = (total_length // self.max_length) * self.max_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + self.max_length] for i in range(0, total_length, self.sequence_stride)]
            for k, t in concatenated_examples.items()
        }
        return result

    def group_texts_within_documents(self, example: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        total_length = len(example["input_ids"])
        doc_result = {
            "input_ids": [],
            "attention_mask": [],
            "special_tokens_mask": []
        }

        # If the length of input is less than `self.max_length`, then add the whole input into `doc_result`.
        if total_length < self.max_length:
            doc_result["input_ids"].append(example["input_ids"])
            doc_result["attention_mask"].append(example["attention_mask"])
            doc_result["special_tokens_mask"].append(example["special_tokens_mask"])
        else:
            doc_result["input_ids"] = [example["input_ids"][i : i + self.max_length]
                                       for i in range(0, total_length, self.sequence_stride)]
            doc_result["attention_mask"] = [example["attention_mask"][i : i + self.max_length]
                                            for i in range(0, total_length, self.sequence_stride)]
            doc_result["special_tokens_mask"] = [example["special_tokens_mask"][i : i + self.max_length]
                                                 for i in range(0, total_length, self.sequence_stride)]
        assert all([len(x) > 0 for x in doc_result["input_ids"]])
        return doc_result

    def encode_with_truncation(self, examples):
        """Mapping function to tokenize the sentences passed with truncation"""
        return self.tokenizer(examples["text"], truncation=True, padding="max_length",
                              max_length=self.max_length, return_special_tokens_mask=True)

    def encode_without_truncation(self, examples):
        """Mapping function to tokenize the sentences passed without truncation"""
        return self.tokenizer(examples["text"], return_special_tokens_mask=True)
