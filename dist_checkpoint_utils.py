import math
from typing import Iterator, TypeVar

import torch
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful


_T_co = TypeVar("_T_co", covariant=True)


class AppState(Stateful):
    """
    Application state to checkpoint in a distributed setting.
    Reference: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
    """
    def __init__(self, model, optimizer, lr_scheduler=None, grad_scaler=None, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.grad_scaler = grad_scaler
        self.extra_state = kwargs  # any extra state to save (counters, etc.)

    def state_dict(self):
        # Get model and optimizer state dicts, handling FSDP
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        state = {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
        }
        if self.lr_scheduler is not None:
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
        if self.grad_scaler is not None:
            state["grad_scaler"] = self.grad_scaler.state_dict()
        state["extra_state"] = self.extra_state
        return state

    def load_state_dict(self, state_dict):
        # Set the model and optimizer state dicts
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )
        if self.lr_scheduler is not None and "lr_scheduler" in state_dict:
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        if self.grad_scaler is not None and "grad_scaler" in state_dict:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
        self.extra_state.update(state_dict["extra_state"])


class StatefulDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Simple wrapper on top of the standard distributed sampler that supports setting up the iterator within the epoch.
    This is important for reloading intermediate model state that requires resuming dataloader from a specific training step.
    Reference: https://github.com/facebookresearch/vissl/blob/09270ed25a6c2cf71263d955b64cbe076d34ac45/vissl/data/data_helper.py
    """
    def __init__(self, dataset: torch.utils.data.dataset.Dataset, batch_size: int, **kwargs):
        super().__init__(dataset, **kwargs)

        # New vars to support restoring dataloader state
        self.batch_size = batch_size
        self.start_iter = None

    def set_start_iter(self, start_iter: int):
        """
        !! This is main new function introduced for resuming model training.
        Set the iteration number from which the sampling should start. This is
        used to find the marker in the data permutation order from where the
        sampler should start sampling.
        """
        start_index = start_iter * self.batch_size
        assert start_index < self.num_samples, f"{start_index} < {self.num_samples} (iterator={start_iter})"
        self.start_iter = start_iter

    def __iter__(self) -> Iterator[_T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # Stateful addition: https://github.com/facebookresearch/vissl/blob/09270ed25a6c2cf71263d955b64cbe076d34ac45/vissl/data/data_helper.py#L93
        if self.start_iter is not None:  # New diff: reload the dataloader state
            start_index = self.start_iter * self.batch_size
            assert start_index < self.num_samples, f"{start_index} < {self.num_samples}"
            indices = indices[start_index:]

        return iter(indices)
