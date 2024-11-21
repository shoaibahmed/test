"""
Prefetcher mainly adapted from:
https://github.com/NVIDIA/apex/blob/70018365c8add3e46574b897149db5d6dd21ef5c/examples/imagenet/main_amp.py#L265
"""

import torch


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.data_dict = next(self.loader)
        except StopIteration:
            self.data_dict = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for k in self.data_dict:
                self.data_dict[k] = self.data_dict[k].cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data_dict = self.data_dict
        if data_dict is not None:
            for k in data_dict:
                data_dict[k].record_stream(torch.cuda.current_stream())
        self.preload()
        return data_dict
