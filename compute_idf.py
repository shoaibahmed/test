#!/bin/python3

import os
from tqdm import tqdm
from argparse import Namespace

import torch

from multiscale_trainer import load_model
from dataset import NLPDataset, get_dataloader


args = Namespace(model_name="mistral", model_size="7b", use_instruct_model=False, amp_dtype=None,
                 use_gradient_checkpointing=False, batch_size=32, seq_length=2048)

# Load the tokenizer
tokenizer = load_model(args, only_tokenizer=True)
vocab_size = len(tokenizer)
print("Vocab size:", vocab_size)

processed_output_file = f"./datasets/fineweb_edu_10b_model_{args.model_name}_idf.pth"
ds_path = f"./datasets/fineweb_edu_model_{args.model_name}_seq_len_2048/"
assert os.path.exists(ds_path)

if not os.path.exists(processed_output_file):
    # Load the dataset
    print("IDF file not found. Loading dataset:", ds_path)
    dataset = NLPDataset.load_dataset(ds_path)  # returns a dataset dict
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    assert test_dataset is None

    print("# train:", len(train_dataset))
    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=False)

    bos_positions = []
    document_lengths = []
    token_counts = []
    prev_bos_position = None  # Keep track of the last BOS position

    # Define the distribution
    cumulative_dist = torch.zeros(vocab_size, dtype=torch.float32)
    seq_count = torch.zeros(vocab_size, dtype=torch.float32)
    weights = torch.ones(args.seq_length, dtype=torch.float32)

    num_seqs = 0
    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"]
        for b in range(len(input_ids)):  # Compute the histogram of tokens
            document_dist = torch.zeros(vocab_size, dtype=torch.float32)
            document_dist.scatter_add_(dim=0, index=input_ids[b], src=weights)
            cumulative_dist += document_dist  # add the cumulative count
            seq_count += (document_dist > 0).float()  # count the number of occurances
            num_seqs += 1

    idf = torch.log((1 + num_seqs) / (1 + seq_count))
    output_dict = dict(cumulative_dist=cumulative_dist, seq_count=seq_count, num_seqs=num_seqs, idf=idf)
    torch.save(output_dict, processed_output_file)
    print("!! Stats saved to file:", processed_output_file)

print("Loading processed file:", processed_output_file)
output_dict = torch.load(processed_output_file)
cumulative_dist = output_dict["cumulative_dist"]
seq_count = output_dict["seq_count"]
idf = output_dict["idf"]
num_seqs = output_dict["num_seqs"]
print(f"IDF: {idf.shape} / cumulative dist: {cumulative_dist.shape} / seq count: {seq_count.shape} / # sequences: {num_seqs}")

k = 256
print("="*100)
vals, idxs = torch.topk(idf, k=k)
print("Top-k tokens with highest IDF:", {tokenizer.decode(int(idx)): float(val) for idx, val in zip(idxs, vals)})

print("="*100)
selected_idx = torch.where(seq_count > 0)[0]  # only select tokens with non-zero sequence count
vals, idxs = torch.topk(idf[selected_idx], k=k, largest=True)
idxs = selected_idx[idxs]  # map the idx back to the original indices
print("Top-k tokens with highest IDF (non zero count):", {tokenizer.decode(int(idx)): float(val) for idx, val in zip(idxs, vals)})

print("="*100)
vals, idxs = torch.topk(idf, k=k, largest=False)
print("Top-k tokens with lowest IDF:", {tokenizer.decode(int(idx)): float(val) for idx, val in zip(idxs, vals)})

print("="*100)
sorted_vals, idxs = torch.sort(idf, stable=True)
for idx, val in zip(idxs, sorted_vals):
    print({"idx": int(idx), "idf": float(val), "count": int(seq_count[idx]), "cumulative_count": int(cumulative_dist[idx]),
           "decoded_token": tokenizer.decode(int(idx))})
