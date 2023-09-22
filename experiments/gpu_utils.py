import torch
import torch.distributed as dist

def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def gpu_available() -> bool:
    return torch.cuda.is_available()