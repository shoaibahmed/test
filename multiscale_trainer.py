#!/bin/python

import torch

BOW_MODULE_LOADED = False
try:
    from bow_module import bow_computation, incremental_bow_computation
    BOW_MODULE_LOADED = True
except:
    print("[WARNING] Loading BoW module failed. Falling back to Python implementation of the preprocessor.")

import os
import gc
import re
import time
import json
import random
import argparse
from tqdm import tqdm
from functools import partial
from packaging import version
from typing import Tuple, List, Dict, TextIO

import wandb

import torch
import numpy as np
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import datasets

from dataset import NLPDataset, get_dataloader
from train_utils import get_num_model_params, get_optimizer, get_lr_scheduler
from dist_utils import init_distributed_env, is_main_proc, get_world_size, wait_for_other_procs, convert_to_distributed, \
    reduce_tensor, gather_tensor, convert_state_dict
from fsdp_utils import get_llama_wrapper, get_mistral_wrapper

from dist_checkpoint_utils import AppState, StatefulDistributedSampler
import torch.distributed.checkpoint as dcp

# from transformers.models.llama.modeling_llama import LlamaForCausalLM
# from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from llama_model import LlamaForCausalLM
from mistral_model import MistralForCausalLM

from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import simple_evaluate

from torcheval.metrics.functional.ranking import retrieval_precision


def load_model(args, only_tokenizer=False, pretrained=False):
    # assumes huggingface login: `huggingface-cli login``
    if args.model_name == "llama-2":
        if args.use_instruct_model:
            model_name = f"meta-llama/Llama-2-{args.model_size.lower()}-chat-hf"
        else:
            model_name = f"meta-llama/Llama-2-{args.model_size.lower()}-hf"
    elif args.model_name == "mistral":
        if args.use_instruct_model:
            model_name = f"mistralai/Mistral-{args.model_size.upper()}-Instruct-v0.2"
        else:
            model_name = f"mistralai/Mistral-{args.model_size.upper()}-v0.1"
    else:
        raise RuntimeError(f"Unsupported model: {args.model_name}")
    print("!! Loading model:", model_name)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if only_tokenizer:
        return tokenizer

    # Load the model as well as the tokenizer
    config = AutoConfig.from_pretrained(model_name)
    print("Config:", config)
    kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    if args.amp_dtype is not None:
        kwargs["torch_dtype"] = torch.float32  # AMP will take care of casting
    print("Model precision:", kwargs["torch_dtype"])
    if pretrained:
        print("Using pretrained model...")

    wrap_policy = None
    if args.model_name == "llama-2":
        if not pretrained:
            model = LlamaForCausalLM(config).to(kwargs["torch_dtype"])
        else:
            model = LlamaForCausalLM.from_pretrained(model_name, **kwargs)
        wrap_policy = get_llama_wrapper()
    elif args.model_name == "mistral":
        if not pretrained:
            model = MistralForCausalLM(config).to(kwargs["torch_dtype"])
        else:
            model = MistralForCausalLM.from_pretrained(model_name, **kwargs)
        wrap_policy = get_mistral_wrapper()
    else:
        raise RuntimeError(f"Unsupported model: {args.model_name}")

    if args.use_gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
    assert wrap_policy is not None
    return model, tokenizer, wrap_policy


class CroppedSequenceWrapper(torch.utils.data.Dataset):
    """
    Wrapper over the dataset that discards excess tokens
    """
    def __init__(self, dataset: datasets.Dataset, num_tokens_to_keep: int):
        super().__init__()
        self.dataset = dataset
        self.num_tokens_to_keep = num_tokens_to_keep

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        return {k: example[k][:self.num_tokens_to_keep] for k in example.keys()}


class DatasetTargetPreprocessor(torch.utils.data.Dataset):
    """
    Preprocess the dataset with the option for incremental BoW computation.
    """
    def __init__(self, dataset: datasets.Dataset, vocab_size: int, bos_token_id: int, prediction_heads: List[int],
                 use_incremental_version: bool = False, multihead_token_weighting_scheme: str = "uniform",
                 idf: torch.Tensor = None):
        super().__init__()
        self.dataset = dataset
        self.prediction_heads = prediction_heads
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.use_incremental_version = use_incremental_version
        self.max_pred_horizon = max(self.prediction_heads)
        self.multihead_token_weighting_scheme = multihead_token_weighting_scheme
        self.idf = idf

        if self.multihead_token_weighting_scheme == "idf":
            assert self.idf is not None, "IDF weights should be specified with IDF token weighting scheme"
            assert self.idf.shape == (self.vocab_size,), f"{self.idf.shape} doesn't match with the vocab size: {self.vocab_size}"
        else:
            assert self.idf is None, "IDF should only be passed for the IDF weighting scheme"

        # Define the multihead token weighting scheme
        if self.multihead_token_weighting_scheme in ["uniform", "truncated_exp", "idf"]:
            self.bag_weights = {}
            self.extra_bag_args = {}
            for bag_size in self.prediction_heads:
                if self.multihead_token_weighting_scheme in ["uniform", "idf"]:  # IDF uses the uniform weighting followed by the IDF weighting
                    self.bag_weights[bag_size] = torch.ones(bag_size, dtype=torch.float32) / bag_size  # V
                    p = None  # p is only relevant for truncated exp distribution
                else:
                    assert self.multihead_token_weighting_scheme == "truncated_exp", self.multihead_token_weighting_scheme
                    # Geometric distribution with p computed at CDF = 0.9
                    p = 1 - np.power(0.1, 1/bag_size)
                    x = np.arange(bag_size)
                    weights = p / (1-np.power(1-p, bag_size)) * np.power(1-p, x)
                    self.bag_weights[bag_size] = torch.tensor(weights, dtype=torch.float32)
                self.extra_bag_args[bag_size] = {"p": p}
        else:
            raise NotImplementedError(f"Token weighting scheme {self.multihead_token_weighting_scheme} not implemented!")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        tokenized_input = example["input_ids"]
        assert len(tokenized_input.shape) == 1, tokenized_input.shape
        original_keys = list(example.keys())

        L = len(tokenized_input)
        target_len = L - self.max_pred_horizon

        # BoW computation
        for pred_horizon in self.prediction_heads:
            target_tensor = torch.zeros(target_len, self.vocab_size, dtype=torch.float32)  # L x V
            weight_vector = self.bag_weights[pred_horizon]  # Shape = (pred horizon,)
            p = self.extra_bag_args[pred_horizon]["p"]

            if self.use_incremental_version:
                relative_freq = torch.zeros(self.vocab_size, dtype=torch.float32)  # Cumulative frequency
                bos_masking = False
                w_1, w_W = weight_vector[0], weight_vector[-1]  # define the weights for the first and last element

                for i in range(target_len):
                    start_idx = i + 1
                    last_idx = i + pred_horizon
                    end_idx = last_idx + 1  # since the last index is excluded

                    # Incrementally update frequency
                    removed_token = tokenized_input[i]       # current token (one less than start_idx)
                    added_token = tokenized_input[last_idx]  # last token
                    if i == 0 or removed_token == self.bos_token_id:  # first sequence or masking ended -> recompute the frequency distribution
                        bos_masking = False  # masking ended
                        relative_freq.zero_()  # reinitialize the frequency
                        current_input_chunk = tokenized_input[start_idx:end_idx]
                        bos_mask = current_input_chunk == self.bos_token_id
                        if bos_mask.any():  # document boundaries present
                            current_input_chunk = current_input_chunk.clone()
                            bos_idx_l = torch.where(bos_mask)[0]  # returns a tuple
                            if len(bos_idx_l) > 0:  # masked tokens present
                                current_input_chunk[bos_idx_l[0]:] = self.bos_token_id  # replace all the remaining tokens with BOS
                                bos_masking = True  # increase weight on BOS token with each additional token -- future tokens masked due to BOS

                        relative_freq.scatter_add_(dim=0, index=current_input_chunk, src=weight_vector)
                    else:
                        if added_token == self.bos_token_id:
                            bos_masking = True  # enable BOS masking for future tokens as BOS token is encountered
                        relative_freq[removed_token] -= w_1  # Remove contribution from cumulative for the removed token
                        if self.multihead_token_weighting_scheme == "truncated_exp":  # upweight the rest of the sequence
                            relative_freq *= 1/(1-p)
                        if bos_masking:  # only boost the score for the BOS token as future tokens are masked
                            relative_freq[self.bos_token_id] += w_W  # Add BOS to cumulative
                        else:
                            relative_freq[added_token] += w_W  # Add to cumulative

                    if self.idf is not None:
                        assert relative_freq.shape == self.idf.shape, f"{relative_freq.shape} != {self.idf.shape}"
                        weighted_freq = (relative_freq * self.idf)
                        target_tensor[i] = weighted_freq / weighted_freq.sum()  # V -> LV
                    else:
                        target_tensor[i] = relative_freq  # V -> LV
            else:
                # Non-incremental version
                for i in range(target_len):
                    relative_freq = torch.zeros(self.vocab_size, dtype=torch.float32)  # V
                    start_idx = i + 1
                    end_idx = i + 1 + pred_horizon  # +1 since the last index is excluded

                    # BOS token should be assigned all the mass towards the end of the doc
                    current_input_chunk = tokenized_input[start_idx:end_idx]
                    bos_mask = current_input_chunk == self.bos_token_id
                    if bos_mask.any():  # document boundaries present
                        current_input_chunk = current_input_chunk.clone()
                        bos_idx_l = torch.where(bos_mask)[0]  # returns a tuple
                        if len(bos_idx_l) > 0:
                            current_input_chunk[bos_idx_l[0]:] = self.bos_token_id  # replace all the remaining tokens with BOS

                    relative_freq.scatter_add_(dim=0, index=current_input_chunk, src=weight_vector)
                    if self.idf is not None:
                        assert relative_freq.shape == self.idf.shape, f"{relative_freq.shape} != {self.idf.shape}"
                        weighted_freq = (relative_freq * self.idf)
                        target_tensor[i] = weighted_freq / weighted_freq.sum()  # V -> LV
                    else:
                        target_tensor[i] = relative_freq  # V -> LV

            example[f"target_{pred_horizon}"] = target_tensor

        for k in original_keys:  # crop the input ids
            example[k] = example[k][:target_len]

        return example


class DatasetTargetCPPPreprocessor(DatasetTargetPreprocessor):
    """
    Wrapper over the CPP implementation of the preprocessor using PyTorch CPP extensions
    """
    def __init__(self, dataset: datasets.Dataset, vocab_size: int, bos_token_id: int, prediction_heads: List[int],
                 use_incremental_version: bool = False, multihead_token_weighting_scheme: str = "uniform",
                 idf: torch.Tensor = None):
        super().__init__(dataset=dataset, vocab_size=vocab_size, bos_token_id=bos_token_id,
                         prediction_heads=prediction_heads, use_incremental_version=use_incremental_version,
                         multihead_token_weighting_scheme=multihead_token_weighting_scheme, idf=idf)

        # Setup the values correctly for the CPP interface
        self.bag_weights = [self.bag_weights[pred_horizon] for pred_horizon in self.prediction_heads]
        self.p_values = [self.extra_bag_args[pred_horizon]["p"] for pred_horizon in self.prediction_heads]
        self.p_values = [x if x is not None else -1. for x in self.p_values]  # set default value for CPP interface
        if self.idf is None:  # for CPP extension to work
            self.idf = torch.empty((0,), dtype=torch.float32)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        tokenized_input = example["input_ids"]
        assert len(tokenized_input.shape) == 1, tokenized_input.shape
        original_keys = list(example.keys())

        L = len(tokenized_input)
        target_len = L - self.max_pred_horizon

        if self.use_incremental_version:
            output = incremental_bow_computation(tokenized_input, self.vocab_size, self.bos_token_id, self.prediction_heads,
                                                 self.bag_weights, self.p_values, self.multihead_token_weighting_scheme, self.idf)
        else:
            output = bow_computation(tokenized_input, self.vocab_size, self.bos_token_id, self.prediction_heads, self.bag_weights,
                                     self.multihead_token_weighting_scheme, self.idf)

        for i, pred_horizon in enumerate(self.prediction_heads):
            example[f"target_{pred_horizon}"] = output[i]

        for k in original_keys:  # crop the input ids
            example[k] = example[k][:target_len]

        return example


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, apply_log_softmax: bool = False):
        super().__init__()
        self.apply_log_softmax = apply_log_softmax

    def forward(self, pred_log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred_log_probs.shape == target.shape, f"{pred_log_probs.shape} != {target.shape}"
        if self.apply_log_softmax:
            pred_log_probs = torch.nn.functional.log_softmax(pred_log_probs, dim=-1)
        loss = - (target * pred_log_probs).sum(dim=-1).mean()  # B x L x V
        return loss


def compute_log_probs(logits: torch.Tensor, target_ids: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    # Apply softmax and log to obtain log probabilities from logits (summing original logits would be incorrect)
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    log_probs = torch.gather(log_probs, dim=2, index=target_ids.unsqueeze(-1)).squeeze(-1)
    sequence_log_prob = log_probs.sum(dim=1).cpu().float().numpy()

    # Calculate perplexity
    sequence_length = target_ids.size(-1)
    assert sequence_length > 0, logits
    sequence_perplexity = np.exp(-sequence_log_prob / sequence_length)

    return sequence_perplexity, sequence_log_prob


def compute_loss(model: torch.nn.Module, tokenizer: PreTrainedTokenizer, tokenized_input: torch.Tensor,
                 prediction_heads: List[int], prediction_head_weights: List[float], amp_dtype: torch.dtype,
                 ) -> Tuple[torch.Tensor, Dict[int, float]]:
    """
    DEPRECATED: Use the recent data preprocessing based target computation
    TODO: add support for incremental computation
    """
    loss_fn = CrossEntropyLoss(apply_log_softmax=False)
    vectorized_impl = True

    # Forward prop through the model and compute the loss (w/ AMP)
    prediction_head_losses = None
    with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
        if len(prediction_heads) > 0:
            prediction_head_losses = {}
            all_preds = model(input_ids=tokenized_input, return_all_predictions=True)
            assert len(all_preds) == len(prediction_heads), f"{len(all_preds)} != {len(prediction_heads)}"
            seq_len = tokenized_input.shape[1]
            vocab_size = all_preds[0].shape[-1]

            loss = 0.
            for idx, pred_horizon in enumerate(prediction_heads):
                # Compute the loss based on the relative token frequency
                assert pred_horizon >= 1, pred_horizon
                sequence_loss = 0.
                for i in range(seq_len-1):
                    lm_logits = all_preds[idx][:, i, :]
                    lm_log_probs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
                    relative_freq = torch.zeros(len(tokenized_input), vocab_size).to(tokenized_input.device)  # B x V
                    start_idx = i + 1
                    last_idx = min(i + 1 + pred_horizon, seq_len)

                    # BOS/EOS token should be assigned all the mass towards the end of the doc
                    # Use an additional max pred horizon tokens that are discarded at the end of computation
                    # Effective context length = input len - max pred horizon
                    current_input_chunk = tokenized_input[:, start_idx:last_idx]
                    if (current_input_chunk == tokenizer.bos_token_id).any():  # document boundaries present
                        current_input_chunk = current_input_chunk.clone()
                        bos_idx_b, bos_idx_l = torch.where(current_input_chunk == tokenizer.bos_token_id)  # B x L sequence
                        last_bos_idx_b = None
                        for i, idx in enumerate(bos_idx_b):  # iterate over unique batch indices and set all elements in the sequence beyond the first BOS token to BOS
                            if bos_idx_b != last_bos_idx_b:
                                seq_idx = bos_idx_l[i]
                                current_input_chunk[idx, seq_idx:] = tokenizer.bos_token_id  # replace all of the remaining tokens with BOS
                                last_bos_idx_b = bos_idx_b

                    if vectorized_impl:
                        relative_freq = torch.scatter_add(relative_freq, dim=1, index=current_input_chunk, src=torch.ones_like(relative_freq))
                    else:
                        for j in range(len(current_input_chunk)):
                            relative_freq[:, current_input_chunk[:, j]] += 1

                    relative_freq = relative_freq / relative_freq.sum(dim=1)  # normalize
                    current_loss = loss_fn(lm_log_probs, relative_freq)
                    sequence_loss = sequence_loss + current_loss  # accumulate the loss over all tokens
                head_weight = prediction_head_weights[idx]
                loss = loss + head_weight * sequence_loss  # accumulate in the total loss for all prediction heads
                prediction_head_losses[pred_horizon] = float(sequence_loss)
        else:
            loss = model(input_ids=tokenized_input, labels=tokenized_input).loss

    return loss, prediction_head_losses


def compute_loss_from_target(model: torch.nn.Module, tokenized_input: torch.Tensor, tokenized_targets: Dict[int, torch.Tensor],
                             prediction_heads: List[int], prediction_head_weights: List[float], amp_dtype: torch.dtype,
                             return_cross_head_outputs: bool = False) -> Tuple[torch.Tensor, Dict[int, float]]:
    prediction_head_losses = None
    loss_fn = CrossEntropyLoss(apply_log_softmax=False)

    # Forward prop through the model and compute the loss (w/ AMP)
    with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
        if len(prediction_heads) > 0:
            prediction_head_losses = {}
            all_preds = model(input_ids=tokenized_input, return_all_predictions=True)
            assert len(all_preds) == len(prediction_heads), f"{len(all_preds)} != {len(prediction_heads)}"

            loss = 0.
            for idx, pred_horizon in enumerate(prediction_heads):
                assert pred_horizon >= 1, pred_horizon
                assert pred_horizon in tokenized_targets, tokenized_targets.keys()
                assert tokenized_targets[pred_horizon].shape[:2] == tokenized_input.shape, \
                    f"{tokenized_targets[pred_horizon].shape[:2]} != {tokenized_input.shape}"

                # Get the model log probs
                lm_log_probs = torch.nn.functional.log_softmax(all_preds[idx], dim=-1)  # B x L x V

                if return_cross_head_outputs:
                    for target_idx, target_pred_horizon in enumerate(prediction_heads):
                        current_targets = tokenized_targets[target_pred_horizon]  # B x L x V
                        sequence_loss = loss_fn(lm_log_probs, current_targets)

                        if idx == target_idx:  # accumulate only matching loss
                            assert pred_horizon == target_pred_horizon, f"{pred_horizon} != {target_pred_horizon}"
                            head_weight = prediction_head_weights[idx]
                            loss = loss + head_weight * sequence_loss  # accumulate in the total loss for all prediction heads

                            # Compute the precision of retrieval
                            # https://pytorch.org/torcheval/main/generated/torcheval.metrics.functional.retrieval_precision.html
                            # Expects input of size, [num_tasks, retrieval_size] and outputs [num_tasks]
                            flattened_targets = (current_targets > 0).reshape(-1, current_targets.shape[-1]).to(lm_log_probs.dtype)
                            precision = retrieval_precision(lm_log_probs.reshape(-1, lm_log_probs.shape[-1]),
                                                            flattened_targets, k=pred_horizon, num_tasks=len(flattened_targets))
                            prediction_head_losses[f"precision_head_{pred_horizon}"] = float(precision.mean())
                        prediction_head_losses[f"pred_{pred_horizon}_target_{target_pred_horizon}"] = float(sequence_loss)
                else:
                    current_targets = tokenized_targets[pred_horizon]  # B x L x V
                    sequence_loss = loss_fn(lm_log_probs, current_targets)

                    head_weight = prediction_head_weights[idx]
                    loss = loss + head_weight * sequence_loss  # accumulate in the total loss for all prediction heads
                    prediction_head_losses[pred_horizon] = float(sequence_loss)
        else:
            loss = model(input_ids=tokenized_input, labels=tokenized_input).loss

    return loss, prediction_head_losses


def train(model: torch.nn.Module, tokenizer: PreTrainedTokenizer, train_loader: torch.utils.data.DataLoader,
          eval_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
          lr_scheduler: torch.optim.lr_scheduler.LRScheduler, prediction_heads: List[int],
          prediction_head_weights: List[float], train_steps: int, eval_after_steps: int,
          gradient_accumulation_steps: int, checkpoint_after_steps: int, device: torch.device,
          amp_dtype: torch.dtype, grad_scaler: torch.cuda.amp.grad_scaler.GradScaler,
          clip_grad_norm: float, checkpoint_file: str, use_lora: bool, file_logger: TextIO):
    pbar = None
    if is_main_proc():
        total_steps = train_steps * gradient_accumulation_steps
        pbar = tqdm(total=total_steps)

    world_size = get_world_size()
    epoch = 0
    iterator = 0  # counts the number of iterations for the loop
    train_step = 0  # counts the optimization steps
    tokens_seen = 0
    last_eval_step = None
    last_checkpoint_step = None
    iterator_within_epoch = None
    training_completed = False
    start_time = time.time()

    checkpoint_state_dir = None
    resume_iterator = False
    if checkpoint_file is not None:
        # Define the checkpoint state path
        checkpoint_path, checkpoint_name = os.path.split(checkpoint_file)
        checkpoint_name = os.path.splitext(checkpoint_name)[0]  # remove the extension
        checkpoint_state_dir = os.path.join(checkpoint_path, f"state_{checkpoint_name}")
        print("Checkpoint state directory:", checkpoint_state_dir)

        checkpoint_saved_marker_file = os.path.join(checkpoint_state_dir, "processed.log")
        if os.path.exists(checkpoint_saved_marker_file):  # Load if the intermediate state exists
            print("!! Found existing checkpoint state. Loading...")
            app_state = AppState(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, grad_scaler=grad_scaler, train_step=0, epoch=0,
                                 iterator=0, iterator_within_epoch=0, tokens_seen=0, last_eval_step=None, last_checkpoint_step=None)
            state_dict = {"app_state": app_state}

            # Load the checkpoint
            dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_state_dir)

            # Extract the counters from app_state.extra_state
            train_step = app_state.extra_state['train_step']
            epoch = app_state.extra_state['epoch']
            iterator = app_state.extra_state['iterator']
            iterator_within_epoch = app_state.extra_state['iterator_within_epoch']
            tokens_seen = app_state.extra_state['tokens_seen']
            last_eval_step = app_state.extra_state['last_eval_step']
            last_checkpoint_step = app_state.extra_state['last_checkpoint_step']

            wait_for_other_procs()  # make sure all processes have loaded the model state
            print(f"!! Resuming training / train step: {train_step} / epoch: {epoch} / iterator: {iterator} / iterator within epoch: {iterator_within_epoch} / tokens seen: {tokens_seen}")

            if pbar is not None:  # Update pbar such that it starts from iterator
                pbar.update(iterator)
            resume_iterator = True
        else:
            app_state = None  # Will create a new one later when saving

    model.train()
    optimizer.zero_grad()

    while True:  # restart at the end of trainer
        if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
            print(f"Setting sampler epoch: {epoch}")
            train_loader.sampler.set_epoch(epoch)

        if resume_iterator:
            assert iterator_within_epoch is not None
            if not hasattr(train_loader, "sampler"):
                raise NotImplementedError(f"Resuming dataloader state without sampler is not yet implemented!")
            print(f"Setting sampler start iteration: {iterator_within_epoch}")
            assert hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, StatefulDistributedSampler), train_loader.sampler
            train_loader.sampler.set_start_iter(iterator_within_epoch)
            resume_iterator = False  # set it to false to ensure future iterations over the dataset reinitialize the iterator to 0
        else:
            iterator_within_epoch = 0

        for batch in train_loader:
            tokenized_input = batch["input_ids"].to(device)
            if f"target_{prediction_heads[0]}" in batch:
                tokenized_targets = {k: batch[f"target_{k}"].to(device) for k in prediction_heads}
                loss, prediction_head_losses = compute_loss_from_target(model, tokenized_input, tokenized_targets, prediction_heads,
                                                                        prediction_head_weights, amp_dtype, return_cross_head_outputs=False)
            else:
                loss, prediction_head_losses = compute_loss(model, tokenizer, tokenized_input, prediction_heads, prediction_head_weights,
                                                            amp_dtype)

            # Accumulate gradients
            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            current_lr = None
            if lr_scheduler is not None:
                current_lr = lr_scheduler.get_last_lr()  # returns the last LR
                assert isinstance(current_lr, list)
                current_lr = current_lr[0]  # get the first element (assuming it should be the same)
                assert isinstance(current_lr, float)

            if iterator % gradient_accumulation_steps == gradient_accumulation_steps - 1:
                if grad_scaler is not None:
                    if clip_grad_norm is not None:
                        # https://pytorch.org/docs/master/notes/amp_examples.html#gradient-clipping
                        grad_scaler.unscale_(optimizer)  # get the gradients in the original scale
                        if isinstance(model, FSDP):
                            model.clip_grad_norm_(clip_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    grad_scaler.step(optimizer)  # won't unscale if already unscaled
                    grad_scaler.update()
                else:
                    if clip_grad_norm is not None:  # clip the gradients before update -- applied on scaled gradients for AMP
                        if isinstance(model, FSDP):
                            model.clip_grad_norm_(clip_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()
                if lr_scheduler is not None:  # LR scheduler update
                    lr_scheduler.step()
                train_step += 1

            iterator += 1
            iterator_within_epoch += 1
            tokens_seen += int(np.prod(batch["input_ids"].shape)) * world_size  # each process separately processes this many tokens

            if pbar is not None:
                output_str = f"loss: {float(loss):.4f}"
                if len(prediction_head_losses) > 0:
                    prediction_head_losses = {k: round(v, 4) for k, v in prediction_head_losses.items()}
                    output_str += f" / prediction heads: {str(prediction_head_losses)}"
                if current_lr is not None:
                    output_str += f" / LR: {current_lr}"
                pbar.set_description(output_str)
                pbar.update(1)
            if wandb.run is not None or file_logger is not None:
                output_dict = {"train_loss": float(loss), "grad_steps": iterator, "optim_steps": train_step, "tokens_seen": tokens_seen}
                if len(prediction_head_losses) > 0:
                    output_dict["prediction_head_losses"] = prediction_head_losses
                if current_lr is not None:
                    output_dict["lr"] = current_lr
                if wandb.run is not None:
                    wandb.log(output_dict)
                if file_logger is not None:
                    file_logger.write(f"{json.dumps(output_dict)}\n")
                    file_logger.flush()
            if eval_after_steps is not None and train_step % eval_after_steps == eval_after_steps - 1 and last_eval_step != train_step:
                print("Evaluating model...")
                evaluate_model(model, tokenizer, eval_loader, prediction_heads, prediction_head_weights, device, amp_dtype, "train", file_logger)
                model.train()
                last_eval_step = train_step

            if train_step >= train_steps:
                print(f"Training completed for {train_steps} steps. Stopping trainer.")
                training_completed = True
                break

            if checkpoint_state_dir is not None and checkpoint_after_steps is not None and train_step != 0 and \
                train_step % checkpoint_after_steps == 0 and last_checkpoint_step != train_step:
                # Note that all processes writes their part of the state dict, so no need to check for main proc
                print("Saving training state...")
                if is_main_proc() and os.path.exists(checkpoint_saved_marker_file):
                    os.remove(checkpoint_saved_marker_file)  # remove previous marker

                extra_state = {'train_step': train_step, 'epoch': epoch, 'iterator': iterator, 'iterator_within_epoch': iterator_within_epoch,
                               'tokens_seen': tokens_seen, 'last_eval_step': last_eval_step, 'last_checkpoint_step': last_checkpoint_step}
                app_state = AppState(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, grad_scaler=grad_scaler, **extra_state)
                state_dict = {"app_state": app_state}

                # Save the checkpoint
                dcp.save(state_dict=state_dict, checkpoint_id=checkpoint_state_dir)

                if is_main_proc():
                    with open(checkpoint_saved_marker_file, "w") as f:  # write the log as a indicator that dataset processing completed
                        f.write("done")

                last_checkpoint_step = train_step
                wait_for_other_procs()  # make sure all processes have saved the model state
                print("Checkpoint state saved:", checkpoint_state_dir)

        if training_completed:
            break
        epoch += 1

    time_elapsed_h = (time.time() - start_time) / (60 * 60)  # convert seconds into hours
    epochs_completed = train_step / len(train_loader)
    print(f"Model training finished / time elapsed: {time_elapsed_h:.2f}h / epochs completed: {epochs_completed:.2f} (counter: {epoch})")

    # Save the final checkpoint
    cpu_state = None
    if isinstance(model, FSDP):
        assert not use_lora
        print("Saving FSDP state dict...")
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = model.state_dict()
    if is_main_proc() and checkpoint_file is not None:  # Save the final model
        if use_lora:
            base_module = model.module if hasattr(model, 'module') else model
            base_module.save_pretrained(checkpoint_file)  # checkpoint file specifies a directory
        else:
            torch.save(cpu_state if cpu_state is not None else model.state_dict(), checkpoint_file)
            print("Model state dict saved:", checkpoint_file)
    wait_for_other_procs()  # wait for the main process to finish writing the final checkpoint


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, tokenizer: PreTrainedTokenizer, eval_loader: torch.utils.data.DataLoader,
                   prediction_heads: List[int], prediction_head_weights: List[float], device: torch.device,
                   amp_dtype: torch.dtype, split_name: str, file_logger: TextIO) -> Tuple[float, Dict[str, float]]:
    model.eval()
    avg_loss = 0.
    prediction_head_losses = None
    num_ex = 0

    for batch in tqdm(eval_loader):
        tokenized_input = batch["input_ids"].to(device)
        if f"target_{prediction_heads[0]}" in batch:
            tokenized_targets = {k: batch[f"target_{k}"].to(device) for k in prediction_heads}
            loss, prediction_head_losses_current = compute_loss_from_target(model, tokenized_input, tokenized_targets,
                                                                            prediction_heads, prediction_head_weights,
                                                                            amp_dtype, return_cross_head_outputs=True)
        else:
            loss, prediction_head_losses_current = compute_loss(model, tokenizer, tokenized_input, prediction_heads,
                                                                prediction_head_weights, amp_dtype)

        avg_loss += float(loss)
        num_ex += len(tokenized_input)
        if prediction_head_losses is None:
            prediction_head_losses = {k: 0. for k in prediction_head_losses_current}
        for k in prediction_head_losses:
            prediction_head_losses[k] += prediction_head_losses_current[k]

    # Collect the stats from all processes
    for k in prediction_head_losses:
        prediction_head_losses[k] = float(reduce_tensor(torch.tensor(prediction_head_losses[k]).to(device)))
    avg_loss = float(reduce_tensor(torch.tensor(avg_loss).to(device)))
    num_ex = int(reduce_tensor(torch.tensor(num_ex).to(device)))

    avg_loss = avg_loss / num_ex
    for k in prediction_head_losses:
        prediction_head_losses[k] = prediction_head_losses[k] / num_ex

    output_dict = {f"eval_{split_name}": {"num_ex": num_ex, "avg_loss": avg_loss, "prediction_head_losses": prediction_head_losses}}
    print(json.dumps(output_dict))
    if file_logger is not None:
        file_logger.write(f"{json.dumps(output_dict)}\n")
        file_logger.flush()
    if split_name is not None and wandb.run is not None:
        wandb.log(output_dict)
    return avg_loss, prediction_head_losses


def main(args):
    init_distributed_env(args)

    # Update the is_main_proc mapping when not using a shared filesystem
    args.shared_fs = not args.no_shared_fs
    print("Using shared FS:", args.shared_fs)
    if args.no_shared_fs:
        global is_main_proc  # important to make sure that the global function name is overridden
        print(f"!! Updating the is_main_proc mapping with local rank ({args.local_rank}) and shared fs info ({args.shared_fs})...")
        old_output = is_main_proc()
        is_main_proc = partial(is_main_proc, local_rank=args.local_rank, shared_fs=args.shared_fs)
        print(f"!! Was previously a main proc: {old_output} / is now a main proc: {is_main_proc()}")

    generator = None
    if args.seed is not None:  # Set process seed to reduce stochasticity
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(seed=args.seed)
        random.seed(args.seed)
        print("Setting process seed:", args.seed)

        # Generator to seed dataloaders
        generator = torch.Generator()
        generator.manual_seed(args.seed)

    # Prediction args setup
    prediction_heads = args.prediction_heads.split(",")
    prediction_heads = [int(x) for x in prediction_heads]
    prediction_head_weights = args.prediction_head_weights.split(",")
    prediction_head_weights = [float(x) for x in prediction_head_weights]
    assert len(prediction_heads) == len(prediction_head_weights), f"{prediction_heads} != {prediction_head_weights}"
    max_bag_size = max(prediction_heads)
    args.sequence_stride = args.sequence_length

    args.retain_sequence_length = None
    if args.preprocessed_dataset_path is not None:
        print("!! Specified preprocessed dataset path:", args.preprocessed_dataset_path)
        _, ds_name = os.path.split(args.preprocessed_dataset_path)

        # Extract seq length and stride from the preprocessed dataset
        pattern = r"seq_len_([0-9]+).*?stride_([0-9]+)"
        match = re.search(pattern, ds_name)
        assert match, match
        ds_seq_len = int(match.group(1))
        ds_stride = int(match.group(2))
        print(f"Preprocessed dataset / sequence length: {ds_seq_len} / stride: {ds_stride}")

        # New dataset should match the stride of the current model, while seq_len can be larger -- need to discard those tokens later
        if args.use_disjoint_sequences:
            assert ds_seq_len == ds_stride, f"{ds_seq_len} == {ds_stride} for disjoint sequences"
            assert ds_seq_len == args.sequence_length, f"Selected dataset seq length {ds_seq_len} is not compatible with the sequence length of the current model {args.sequence_length}"
        else:
            assert ds_seq_len >= ds_stride, f"{ds_seq_len} should be >= {ds_stride}"
            assert ds_stride == args.sequence_length, f"Selected dataset stride {ds_stride} is not compatible with the sequence length of the current model {args.sequence_length}"
            excess_tokens = (ds_seq_len - ds_stride) - max_bag_size  # tokens that should be removed from the dataset
            if excess_tokens > 0:
                args.retain_sequence_length = ds_seq_len - excess_tokens
            print(f"Preprocessed dataset / excess tokens: {excess_tokens} / retain sequence length: {args.retain_sequence_length}")
        args.dataset_output_dir = args.preprocessed_dataset_path  # replace the directory path
        dataset_dir = f"{args.dataset}_model_{args.model_name}_seq_len_{args.sequence_length+max_bag_size}_stride_{args.sequence_length}"
    else:
        if not args.use_disjoint_sequences:
            args.sequence_length = args.sequence_length + max_bag_size
            print(f"!! Using overlapping sequences of size {args.sequence_length} with a stride of {args.sequence_stride}")
        else:
            print(f"!! Using disjoint i.e., non-overlapping sequences of size {args.sequence_length}")

        dataset_dir = f"{args.dataset}_model_{args.model_name}_seq_len_{args.sequence_length}"
        if not args.use_disjoint_sequences:
            dataset_dir = f"{dataset_dir}_stride_{args.sequence_stride}"
        if args.subsample_size is not None:
            dataset_dir = f"{dataset_dir}_subsample_{args.subsample_size}"
        args.dataset_output_dir = os.path.join(args.dataset_base_dir, dataset_dir)
    print("Dataset output directory:", args.dataset_output_dir)

    if is_main_proc() and not os.path.exists(args.checkpoint_base_dir):
        os.mkdir(args.checkpoint_base_dir)
        print("Checkpoint directory created:", args.checkpoint_base_dir)

    if is_main_proc() and not os.path.exists(args.logs_base_dir):
        os.mkdir(args.logs_base_dir)
        print("Logs directory created:", args.logs_base_dir)

    suffix = "pretrained_" if args.use_pretrained_model else ""
    suffix += f"lora_r_{args.lora_rank}_" if args.use_lora else ""
    suffix += f"steps_{args.train_steps}" if args.train_steps is not None else f"epochs_{args.train_epochs}"
    suffix += f"_bs_{args.batch_size}_effective_{args.effective_batch_size}"
    suffix += f"_grad_accum_{args.gradient_accumulation_steps}" if args.gradient_accumulation_steps > 1 else ""
    suffix += f"_lr_{args.learning_rate}"
    suffix += f"_scheduler_{args.lr_scheduler}" if args.lr_scheduler is not None else ""
    suffix += f"_warmup_{args.lr_warmup_steps}" if args.lr_warmup_steps is not None else ""
    suffix += f"_min_lr_{args.min_learning_rate}" if args.min_learning_rate > 0 else ""
    suffix += f"_wd_{args.weight_decay}" if args.weight_decay > 0 else ""
    suffix += f"{('_amp_' + args.amp_dtype + ('_grad_scaler_' if args.use_grad_scaler else ''))}" if args.amp_dtype is not None else ""
    suffix += f"_clip_{args.clip_grad_norm}" if args.clip_grad_norm is not None else ""
    suffix += f"_prediction_heads_{args.prediction_heads}"
    suffix += f"_weights_{args.prediction_head_weights}"
    suffix += f"_only_head" if args.train_only_head else ""
    suffix += f"_token_weights_{args.multihead_token_weighting_scheme}" if args.multihead_token_weighting_scheme != "uniform" else ""
    args.checkpoint_file = os.path.join(args.checkpoint_base_dir, f"checkpoint_{dataset_dir}_{suffix}.pth")
    if args.evaluate_pretrained_model:  # reset args for pretrained model eval
        args.checkpoint_file = None
        suffix = "pretrained"
    if args.use_lora:
        args.checkpoint_file = args.checkpoint_file.replace(".pth", "")  # need to specify directory

    if args.wandb_project is not None and is_main_proc():
        print("Initialization w&b...")
        args.wandb_run_name = f"{dataset_dir}_{suffix}"
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args, resume=False)

    # Setup local log file
    args.log_file = os.path.join(args.logs_base_dir, f"{dataset_dir}_{suffix}.log")
    file_logger = None
    if is_main_proc():
        print("Logging data to local log file:", args.log_file)
        file_logger = open(args.log_file, "w")

    # Update the args directly with head config
    args.prediction_heads = prediction_heads
    args.prediction_head_weights = prediction_head_weights

    if is_main_proc() and not NLPDataset.is_dataset_processed(args.dataset_output_dir):
        tokenizer = load_model(args, only_tokenizer=True)
        dataset = NLPDataset(args.dataset, tokenizer, max_length=args.sequence_length, sequence_stride=args.sequence_stride,
                             combine_documents=True, subsample_size=args.subsample_size)
        dataset.save_datasets(args.dataset_output_dir)
    wait_for_other_procs()  # wait for the main process to write the dataset

    # Load the dataset
    dataset = NLPDataset.load_dataset(args.dataset_output_dir)  # returns a dataset dict
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    if test_dataset is None:
        if args.preprocessed_dataset_path is not None:
            ds_dir, ds_name = os.path.split(args.preprocessed_dataset_path)
            wikitext_dataset_dir = ds_name.replace(args.dataset, "wikitext-2")
            wikitext_dataset_dir = wikitext_dataset_dir.replace(f"_subsample_{args.subsample_size}", "")
            wikitext_dataset_dir = os.path.join(ds_dir, wikitext_dataset_dir)
        else:
            wikitext_dataset_dir = dataset_dir.replace(args.dataset, "wikitext-2")
            wikitext_dataset_dir = wikitext_dataset_dir.replace(f"_subsample_{args.subsample_size}", "")
            wikitext_dataset_dir = os.path.join(args.dataset_base_dir, wikitext_dataset_dir)
        print("Wikitext-2 dataset dir:", wikitext_dataset_dir)
        if is_main_proc() and not NLPDataset.is_dataset_processed(wikitext_dataset_dir):
            tokenizer = load_model(args, only_tokenizer=True)
            test_dataset = NLPDataset("wikitext-2", tokenizer, max_length=args.sequence_length, sequence_stride=args.sequence_stride,
                                      combine_documents=True, subsample_size=None)
            test_dataset.save_datasets(wikitext_dataset_dir)
        wait_for_other_procs()  # wait for the main process to write the dataset
        test_dataset = NLPDataset.load_dataset(wikitext_dataset_dir)  # returns a dataset dict
        test_dataset = test_dataset["train"]

    print("Train set:", len(train_dataset))
    print("Test set:", len(test_dataset))

    idf = None
    if args.multihead_token_weighting_scheme == "idf":  # load the IDF statistics
        idf_stats_file = os.path.join(args.dataset_base_dir, f"{args.dataset}_10b_model_{args.model_name}_idf.pth")
        assert os.path.exists(idf_stats_file), f"File not found: {idf_stats_file}"
        idf = torch.load(idf_stats_file)["idf"]
        print("Loaded IDF stats from file:", idf_stats_file)

    tokenizer = load_model(args, only_tokenizer=True)  # load the tokenizer
    ds_kwargs = dict(bos_token_id=tokenizer.bos_token_id, prediction_heads=args.prediction_heads,
                     multihead_token_weighting_scheme=args.multihead_token_weighting_scheme, idf=idf)

    if args.validate_dataset:
        for i in range(5):
            print(f"Train set sample {i}:", train_dataset[i])
            print(f"Decoded sample {i}:", tokenizer.decode(train_dataset[i]["input_ids"]))

        if args.retain_sequence_length is not None:
            print("!! Adding a sequence cropping wrapper on top of the preprocessed dataset...")
            train_dataset_cropped = CroppedSequenceWrapper(train_dataset, args.retain_sequence_length)
            for i in range(10):
                example = train_dataset[i]
                cropped_example = train_dataset_cropped[i]
                orig_shape = {k: example[k].shape for k in example}
                new_shape = {k: cropped_example[k].shape for k in example}
                print(f"[{i}] original shapes: {orig_shape} / new shapes: {new_shape}")
            train_dataset = train_dataset_cropped

        processed_ds = DatasetTargetPreprocessor(train_dataset, len(tokenizer), **ds_kwargs)
        processed_ds_inc = DatasetTargetPreprocessor(train_dataset, len(tokenizer), use_incremental_version=True, **ds_kwargs)
        if BOW_MODULE_LOADED:
            processed_ds_cpp = DatasetTargetCPPPreprocessor(train_dataset, len(tokenizer), **ds_kwargs)
            processed_ds_inc_cpp = DatasetTargetCPPPreprocessor(train_dataset, len(tokenizer), use_incremental_version=True, **ds_kwargs)
        else:
            print("!! BoW CPP failed to load. Skipping comparison against CPP extension.")

        # Validate correctness
        num_samples = 100
        abs_tol = 1e-4
        for i in tqdm(range(min(num_samples, len(processed_ds)))):
            print(f"[{i}] original: {train_dataset[i]}")
            print(f"[{i}] processed: {processed_ds_inc[i]}")
            print(f"[{i}] input IDs / orig: {train_dataset[i]['input_ids'].shape} / processed: {processed_ds[i]['input_ids'].shape}")
            assert all([torch.allclose(processed_ds[i][k], processed_ds_inc[i][k], atol=abs_tol) for k in processed_ds[i]]), \
                f"{processed_ds[i]} != {processed_ds_inc[i]}"
            if BOW_MODULE_LOADED:
                assert all([torch.allclose(processed_ds[i][k], processed_ds_cpp[i][k], atol=abs_tol) for k in processed_ds[i]]), \
                    f"{processed_ds[i]} != {processed_ds_cpp[i]}"
                assert all([torch.allclose(processed_ds[i][k], processed_ds_inc_cpp[i][k], atol=abs_tol) for k in processed_ds[i]]), \
                    f"{processed_ds[i]} != {processed_ds_inc_cpp[i]}"
        print("Assertion tests completed...")

        # Benchmark performance
        benchmarking_samples = 1000
        loader_list = [(processed_ds, "non_incremental"), (processed_ds_cpp, "cpp_non_incremental")]
        if BOW_MODULE_LOADED:
            loader_list += [(processed_ds_inc, "incremental"), (processed_ds_inc_cpp, "cpp_incremental")]

        for loader, set_name in loader_list:
            print(f"!! Benchmarking {set_name} version...")
            start = time.time()
            for i in tqdm(range(benchmarking_samples)):
                output = loader[i % len(loader)]
            print(f"Elapsed time: {time.time()-start:.2f} seconds")
        print("Dataset testing successfully completed!")
        exit()

    if args.retain_sequence_length is not None:
        print("!! Adding a sequence cropping wrapper on top of the preprocessed dataset...")
        train_dataset = CroppedSequenceWrapper(train_dataset, args.retain_sequence_length)
        test_dataset = CroppedSequenceWrapper(test_dataset, args.retain_sequence_length)

    # Load the model
    model, tokenizer, fsdp_wrap_policy = load_model(args, pretrained=args.use_pretrained_model)

    # Load from the same random init -- save if random init doesn't exist
    if not args.use_pretrained_model:
        args.random_init_checkpoint_file = os.path.join(args.checkpoint_base_dir, f"random_init_{args.model_name}.pth")
        if is_main_proc() and not os.path.exists(args.random_init_checkpoint_file):
            torch.save(model.state_dict(), args.random_init_checkpoint_file)
            print("Random init checkpoint file saved:", args.random_init_checkpoint_file)
        wait_for_other_procs()  # wait for the main proc to save the checkpoint
        print("!! Loading random init checkpoint from:", args.random_init_checkpoint_file)
        model.load_state_dict(torch.load(args.random_init_checkpoint_file, map_location="cpu"))
    wait_for_other_procs()

    assert len(tokenizer) == model.model.vocab_size, f"{len(tokenizer)} != {model.model.vocab_size}"
    num_model_params = get_num_model_params(model)
    print(f"# model params: {num_model_params/1_000_000:.2f}M")
    if args.use_lora:
        print("LoRA rank:", args.lora_rank)
        peft_config = LoraConfig(
            r=args.lora_rank,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)  # convert the model to PeFT model
        num_lora_model_params = get_num_model_params(model)
        print(f"# LoRA model params: {num_lora_model_params/1_000_000:.2f}M")
        model.print_trainable_parameters()  # print trainable parameters directly using the PEFT interface

    # Set the correct number of heads for the model
    if args.train_only_head:
        assert len(args.prediction_heads) > 0
        for param in model.parameters():  # set all the params from the original model to have no gradient
            param.requires_grad = False
    model.set_num_prediction_heads(len(args.prediction_heads))
    if args.train_only_head:
        num_model_params = get_num_model_params(model.get_prediction_heads())
        print(f"# model params in the head: {num_model_params/1_000_000:.2f}M")

    # Convert to DDP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # move to device
    model = convert_to_distributed(model, args.local_rank, use_ddp=args.use_ddp,
                                   find_unused_parameters=args.train_only_head,
                                   use_orig_params=args.compile_model,
                                   fsdp_auto_wrap_policy=fsdp_wrap_policy)  # cast into DDP/FSDP
    base_module = model.module if hasattr(model, 'module') else model

    if args.compile_model:
        if version.parse(torch.__version__) > version.parse("2.0"):
            print("Compiling model...")
            model.compile()  # Use the torch compile method`
        else:
            print("[WARNING] Can't compile model for PyTorch version < 2.")

    if not args.no_target_preprocessing:
        print("!! Wrapping the dataset into the target preprocessor...")
        if BOW_MODULE_LOADED:
            print("Using CPP incremental data pre-processor with weighting scheme:", args.multihead_token_weighting_scheme)
            train_dataset = DatasetTargetCPPPreprocessor(train_dataset, len(tokenizer), use_incremental_version=True, **ds_kwargs)
            test_dataset = DatasetTargetCPPPreprocessor(test_dataset, len(tokenizer), use_incremental_version=True, **ds_kwargs)
        else:
            print("Using python non-incremental data pre-processor with weighting scheme:", args.multihead_token_weighting_scheme)
            train_dataset = DatasetTargetPreprocessor(train_dataset, len(tokenizer), **ds_kwargs)
            test_dataset = DatasetTargetPreprocessor(test_dataset, len(tokenizer), **ds_kwargs)

    # Create the dataloaders
    train_loader = get_dataloader(train_dataset, args.batch_size, args.num_workers, drop_last=True, generator=generator)
    eval_loader = get_dataloader(test_dataset, args.test_batch_size, args.num_workers, generator=generator)

    if args.train_epochs is not None:
        assert args.train_steps is None, args.train_steps
        args.train_steps = int((len(train_loader) * args.train_epochs) / args.gradient_accumulation_steps)
        args.train_steps = min(gather_tensor(args.train_steps))
        print(f"!! Selected epochs: {args.train_epochs} / # examples in dataset: {len(train_loader)} / grad accumulation steps: {args.gradient_accumulation_steps} / total training steps: {args.train_steps}")
    print("Total number of training steps:", args.train_steps)

    # See about page on https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
    task_list = [("mmlu", 5, ["acc,none"]), ("gsm8k", 5, ["exact_match,strict-match"]), ("hellaswag", 10, ["acc_norm,none"]),
                 ("truthfulqa_mc2", 0, ["acc,none"]), ("winogrande", 5, ["acc,none"]), ("arc_easy", 25, ["acc_norm,none"]),
                 ("arc_challenge", 25, ["acc_norm,none"]), ("piqa", 5, ["acc_norm,none"]), ("boolq", 0, ["acc,none"]),
                 ("lambada_standard", 0, ["acc,none", "perplexity,none"]), ("toxigen", 0, ["acc_norm,none"])]

    if not args.evaluate_pretrained_model and not args.no_baseline_eval:  # perform baseline evaluation before model training
        print("Performing baseline evaluation before model training...")
        eval_start_time = time.time()
        evaluate_model(model, tokenizer, eval_loader, args.prediction_heads, args.prediction_head_weights, device,
                       args.amp_dtype, "baseline_loss", file_logger)
        if args.use_harness_baseline_evals:
            if not args.use_ddp:
                print("!! Reinitializing the model separately for harness baseline eval...")
                initial_model, _, _ = load_model(args, pretrained=args.use_pretrained_model)
                initial_model = initial_model.to(device)  # move to device
                if not args.use_pretrained_model:
                    print("!! Loading random init checkpoint from:", args.random_init_checkpoint_file)
                    initial_model.load_state_dict(torch.load(args.random_init_checkpoint_file, map_location="cpu"))
                if args.compile_model:
                    if version.parse(torch.__version__) > version.parse("2.0"):
                        print("Compiling model...")
                        initial_model.compile()  # Use the torch compile method`
                    else:
                        print("[WARNING] Can't compile model for PyTorch version < 2.")
                model_wrapper = HFLM(pretrained=initial_model, tokenizer=tokenizer, backend="causal")
                print(f"Harness HF wrapper / rank: {model_wrapper.rank} / world size: {model_wrapper.world_size}")

            results_list = {}
            metrics_list = {}
            for task, num_fewshot, metric_list in task_list:
                print("-"*50)
                print(f"Evaluating task: {task} / # few shot: {num_fewshot}")
                current_task_list = [task]
                results = simple_evaluate(model=model_wrapper, model_args=None, tasks=current_task_list, batch_size=1,
                                          cache_requests=True, limit=None, num_fewshot=num_fewshot, log_samples=False)
                results_list[task] = results

                if results is not None:  # main proc
                    print(results)
                    current_metric_list = []
                    metric_dict = {}
                    for metric_name in metric_list:
                        metric_val = results["results"][task][metric_name]
                        print(f">> task: {task} / metric name: {metric_name} / metric val: {metric_val}")
                        metric_dict[metric_name.split(',')[0]] = metric_val
                        current_metric_list.append(metric_val)

                    output_dict = {f"baseline_{task}": metric_dict}
                    if wandb.run is not None:
                        wandb.log(output_dict)
                    if file_logger is not None:
                        file_logger.write(f"{json.dumps(output_dict)}\n")
                        file_logger.flush()
                    metrics_list[task] = current_metric_list

            print(f">>>>> Baseline statistics <<<<<")
            print(f"Metrics list: {metrics_list}")
            print("="*25)

            del model_wrapper
            del initial_model
            torch.cuda.empty_cache()
            gc.collect()

        eval_time_elapsed_h = (time.time() - eval_start_time) / (60 * 60)  # convert seconds into hours
        print(f"Baseline model evaluation completed / time elapsed: {eval_time_elapsed_h:.2f}h")

    if not args.evaluate_pretrained_model and not os.path.exists(args.checkpoint_file):
        print(f"Checkpoint file not found: {args.checkpoint_file}")
        print(">> Initiating model training...")

        # Setup optimizer
        print(f"Optimizer name: {args.optimizer_name} / lr: {args.learning_rate} / wd: {args.weight_decay}")
        if args.train_only_head and len(args.prediction_heads) > 1:
            print("Applying optimizer only on the prediction heads...")
            optimizer = get_optimizer(base_module.get_prediction_heads(), lr=args.learning_rate, wd=args.weight_decay, optimizer_name=args.optimizer_name)
        else:
            optimizer = get_optimizer(model, lr=args.learning_rate, wd=args.weight_decay, optimizer_name=args.optimizer_name)

        grad_scaler = None
        if args.amp_dtype is not None:  # convert the amp_dtype to torch dtype
            if args.amp_dtype == "fp16":
                args.amp_dtype = torch.float16
                if not args.use_grad_scaler:
                    print("[WARNING] float16 AMP is being used without GradScaler")
            else:
                assert args.amp_dtype == "bfp16", args.amp_dtype
                args.amp_dtype = torch.bfloat16
            print("Using AMP dtype:", args.amp_dtype)

            if args.use_grad_scaler:
                if args.use_ddp:
                    print("Using gradient scaler...")
                    grad_scaler = torch.cuda.amp.GradScaler()
                else:
                    print("Using FSDP gradient scaler...")
                    grad_scaler = ShardedGradScaler()

        lr_scheduler = None
        if args.lr_scheduler is not None:
            lr_scheduler = get_lr_scheduler(optimizer, args.train_steps, args.learning_rate, args.min_learning_rate, args.lr_warmup_steps,
                                            scheduler_name=args.lr_scheduler)

        # Train the model
        train(model, tokenizer, train_loader, eval_loader, optimizer, lr_scheduler, args.prediction_heads, args.prediction_head_weights,
              args.train_steps, args.eval_after_steps, args.gradient_accumulation_steps, args.checkpoint_after_steps, device, args.amp_dtype,
              grad_scaler, args.clip_grad_norm, args.checkpoint_file, args.use_lora, file_logger)

        wait_for_other_procs()
        print("!! Model training finished...")
        del optimizer
        del grad_scaler
        torch.cuda.empty_cache()
        gc.collect()

    # Remove the model object before reinitialization
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Reload the model to ensure that eval harness works as expected
    print("Reinitializing model for evaluation without DDP/FSDP...")
    model, _, _ = load_model(args, pretrained=args.use_pretrained_model)
    model.set_num_prediction_heads(len(args.prediction_heads))  # set the right prediction heads
    model = model.to(device)  # move to device
    if args.compile_model:
        if version.parse(torch.__version__) > version.parse("2.0"):
            print("Compiling model...")
            model.compile()  # Use the torch compile method`
        else:
            print("[WARNING] Can't compile model for PyTorch version < 2.")
    wait_for_other_procs()  # wait for all other processes to load the checkpoint

    # Define the HF wrapper for lm-eval-harness eval
    model_wrapper = HFLM(pretrained=model, tokenizer=tokenizer, backend="causal")
    print(f"Harness HF wrapper / rank: {model_wrapper.rank} / world size: {model_wrapper.world_size}")

    # Load the final checkpoint
    if not args.evaluate_pretrained_model and args.checkpoint_file is not None:  # Save the final model
        if args.use_lora:  # PEFT only saves the delta
            # FIXME: What's the equivalent of convert_state_dict with PEFT?
            print("Loading PEFT model delta...")
            model = PeftModel.from_pretrained(model, args.checkpoint_file)  # checkpoint file specifies a directory
        else:  # state dict needs to be converted for non-distributed setting in eval mode
            model.load_state_dict(convert_state_dict(torch.load(args.checkpoint_file, map_location="cpu"), require_module=False))
        print("Loaded model from checkpoint:", args.checkpoint_file)

    if not args.no_final_eval:  # baseline eval = final eval for pretrained model
        print("Performing final evaluation...")
        eval_start_time = time.time()
        evaluate_model(model, tokenizer, eval_loader, args.prediction_heads, args.prediction_head_weights, device,
                       args.amp_dtype, "final_loss", file_logger)
        if args.use_harness_evals:
            results_list = {}
            metrics_list = {}
            for task, num_fewshot, metric_list in task_list:
                print("-"*50)
                print(f"Evaluating task: {task} / # few shot: {num_fewshot}")
                current_task_list = [task]
                results = simple_evaluate(model=model_wrapper, model_args=None, tasks=current_task_list, batch_size=1,
                                          cache_requests=True, limit=None, num_fewshot=num_fewshot, log_samples=False)
                results_list[task] = results

                if results is not None:  # main proc
                    print(results)
                    current_metric_list = []
                    metric_dict = {}
                    for metric_name in metric_list:
                        metric_val = results["results"][task][metric_name]
                        print(f">> task: {task} / metric name: {metric_name} / metric val: {metric_val}")
                        metric_dict[metric_name.split(',')[0]] = metric_val
                        current_metric_list.append(metric_val)

                    output_dict = {f"final_{task}": metric_dict}
                    if wandb.run is not None:
                        wandb.log(output_dict)
                    if file_logger is not None:
                        file_logger.write(f"{json.dumps(output_dict)}\n")
                        file_logger.flush()
                    metrics_list[task] = current_metric_list

            print(f">>>>> Final statistics <<<<<")
            print(f"Metrics list: {metrics_list}")
            print("="*25)
        eval_time_elapsed_h = (time.time() - eval_start_time) / (60 * 60)  # convert seconds into hours
        print(f"Final model evaluation completed / time elapsed: {eval_time_elapsed_h:.2f}h")

    if wandb.run is not None:
        wandb.finish()
    if file_logger is not None:
        file_logger.close()
    wait_for_other_procs()  # introduce a barrier to ensure all processing is finished before the destruction of the process group

    if torch.distributed.is_initialized():  # cleanup the process group
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    supported_datasets = ['fineweb_edu', 'pg19', 'cc_news', 'wikitext-2', 'bookcorpus', 'c4', 'openwebtext', 'slimpajama']
    multihead_token_weighting_schemes = ['uniform', 'truncated_exp', 'idf']

    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Argument parser for multi-scale LLM trainer')

    # Add arguments
    parser.add_argument('--dataset-base-dir', type=str, default='./datasets/',
                        help='Directory to be used for storing processed datasets')
    parser.add_argument('--preprocessed-dataset-path', type=str, default=None,
                        help='Path to preprocessed dataset (default path is used if unspecified)')
    parser.add_argument('--checkpoint-base-dir', type=str, default='./checkpoints/',
                        help='Directory to be used for storing model checkpoints')
    parser.add_argument('--logs-base-dir', type=str, default='./logs/',
                        help='Directory to be used for storing model logs')
    parser.add_argument('-d', '--dataset', default='wikitext-2', choices=supported_datasets,
                        help='Dataset name (default: wikitext-2)')
    parser.add_argument('-m', '--model-name', default='llama-2', choices=['llama-2', 'mistral'],
                        help='Model name (default: llama-2)')
    parser.add_argument('-s', '--model-size', default='7b', choices=['7b'],
                        help='Model size (default: 7b)')
    parser.add_argument('--use-instruct-model', action='store_true', default=False,
                        help='Use instruction-tuned model rather than the base model')
    parser.add_argument('--compile-model', action='store_true', default=False,
                        help='Compile model (only applicable for PyTorch > 2.0)')
    parser.add_argument('--use-pretrained-model', action='store_true', default=False,
                        help='Use pretrained model (instead of training from scratch)')
    parser.add_argument('--use-lora', action='store_true', default=False,
                        help='Use LoRA for finetuning instead of directly finetuning the model')
    parser.add_argument('--lora-rank', type=int, default=8,
                        help='LoRA rank to be used')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size per process (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        help='Batch size per process for testing (default: equal to --batch-size)')
    parser.add_argument('--train-steps', type=int, default=None,
                        help='Number of training steps -- mutually exclusive with --train-epochs (default: None)')
    parser.add_argument('--train-epochs', type=int, default=None,
                        help='Number of training epochs -- mutually exclusive with --train-steps (default: None)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Number of gradient steps to accumulate before calling optimizer.step()')
    parser.add_argument('--amp-dtype', default='None', choices=['None', 'fp16', 'bfp16'],
                        help='AMP dtype for model training (defaults to None i.e., no AMP)')
    parser.add_argument('--use-grad-scaler', action='store_true', default=False,
                        help='Use gradient scaler for training (useful when using AMP)')
    parser.add_argument('--eval-after-steps', type=int, default=None,
                        help='Evaluate the model after the specified number of optimization steps (default: None)')
    parser.add_argument('--checkpoint-after-steps', type=int, default=None,
                        help='Checkpoint the model after the specified number of optimization steps (default: None)')
    parser.add_argument('--evaluate-pretrained-model', action='store_true', default=False,
                        help='Perform evaluation of the pretrained model (w/o any model finetuning)')
    parser.add_argument('--enable-train-eval', action='store_true', default=False,
                        help='Perform final evaluation on the training set (in addition to the test set)')
    parser.add_argument('--no-baseline-eval', action='store_true', default=False,
                        help='No evaluation before model training (eval important for pretrained models)')
    parser.add_argument('--no-final-eval', action='store_true', default=False,
                        help='No final evaluation after model training')
    parser.add_argument('--use-ddp', action='store_true', default=False,
                        help='Use DDP instead of FSDP')
    parser.add_argument('--dist-socket-timeout', type=int, default=1,
                        help='Timeout in hours for the distributed environment -- set it up to a large value for dataset preprocessing to avoid timeout error (default: 1)')
    parser.add_argument('--use-gradient-checkpointing', action='store_true', default=False,
                        help='Perform final evaluation on the training set (in addition to the test set)')
    parser.add_argument('--sequence-length', type=int, default=1024,
                        help='Sequence length for computing the model perplexity (default: 1024)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='learning rate for optimization (default: 1e-4)')
    parser.add_argument('--min-learning-rate', type=float, default=1e-8,
                        help='Minimum learning rate (default: 1e-8)')
    parser.add_argument('--lr-warmup-steps', type=int, default=1000,
                        help='number of LR warmup steps (default: 1000)')
    parser.add_argument('--lr-scheduler', default="cosine", choices=["none", "cosine", "linear"],
                        help='LR scheduler to be used for training (default: cosine)')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                        help='weight decay for optimization (default: 0.1)')
    parser.add_argument('--clip-grad-norm', type=float, default=None,
                        help='gradient clipping magnitude (default: None)')
    parser.add_argument('--optimizer-name', default='adamw', choices=['adamw', 'adam', 'sgd'],
                        help='optimizer name (default: adamw)')
    parser.add_argument('--subsample-size', type=int, default=1000000,
                        help='Dataset subsample size in terms of number of docs (default: 1M)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers for the dataloader (default: 8)')
    parser.add_argument('--seed', type=int, default=43,
                        help='seed value (default: 43)')
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='W&B project name (none indicates no W&B initialization)')
    parser.add_argument('--validate-dataset', action='store_true', default=False,
                        help='Debugging flag which decodes and visualizes examples from the preprocessed dataset')
    parser.add_argument('--prediction-heads', type=str, default='1',
                        help='Prediction heads to be used for model training')
    parser.add_argument('--prediction-head-weights', type=str, default='1',
                        help='Weights to be assigned to the different predictions heads for model training')
    parser.add_argument('--train-only-head', action='store_true', default=False,
                        help='Train only the prediction head while keeping the model fixed')
    parser.add_argument('--use-harness-baseline-evals', action='store_true', default=False,
                        help='Use eval-harness for reporting baseline performance')
    parser.add_argument('--use-harness-evals', action='store_true', default=False,
                        help='Use eval-harness for reporting performance')
    parser.add_argument('--no-target-preprocessing', action='store_true', default=False,
                        help='Process the target directly at prediction time without preprocessing it')
    parser.add_argument('--multihead-token-weighting-scheme', default='uniform', choices=multihead_token_weighting_schemes,
                        help='Token weighting scheme')
    parser.add_argument('--no-shared-fs', action='store_true', default=False,
                        help='Assume separate filesystem per node, necessitating the use of local rank for the is_main_proc evals')
    parser.add_argument('--use-disjoint-sequences', action='store_true', default=False,
                        help=('Uses disjoint i.e., non-overlapping sequences of size {context_window} for training. '
                              'Otherwise, the system packs an additional {max_bag_size} tokens which are discarded after preprocessing. '
                              'Therefore, the model is trained on {context_Window} tokens instead of {context_window-max_bag_size} when using disjoint sequences.'))

    # Parse the arguments
    args = parser.parse_args()

    assert (args.train_steps is None and args.train_epochs is not None) or (args.train_steps is not None and args.train_epochs is None), \
        "--train-epochs and --train-steps are mutually exclusion, but spcification of one is required"
    assert not args.use_lora or args.use_pretrained_model, "Use of LoRA w/o pretrained model is incorrect"
    assert not args.use_lora or args.use_ddp, "FSDP for LoRA is not preferred"
    assert len(args.prediction_heads.split(",")) == len(args.prediction_head_weights.split(",")), \
        f"list length mismatch: {args.prediction_heads.split(',')} != {args.prediction_head_weights.split(',')}"

    if args.amp_dtype == "None":
        args.amp_dtype = None
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size
        print("Setting test batch size to be equal to batch size:", args.test_batch_size)
    if args.subsample_size <= 0:
        args.subsample_size = None
    if args.lr_warmup_steps <= 0:
        args.lr_warmup_steps = None
    if args.lr_scheduler == "none":
        args.lr_scheduler = None
    if args.eval_after_steps <= 0:
        args.eval_after_steps = None
    if args.checkpoint_after_steps <= 0:
        args.checkpoint_after_steps = None
    if args.preprocessed_dataset_path in ["", "none", "None"]:
        args.preprocessed_dataset_path = None

    main(args)
