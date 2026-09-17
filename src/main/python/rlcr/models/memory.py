"""Release unused device allocations between sequential model stages."""
import gc
import torch


def clear_device_cache():
    gc.collect()
    torch.cuda.empty_cache()
