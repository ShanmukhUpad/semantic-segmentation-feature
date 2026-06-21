"""Device selection helper so the same code runs on CPU and on a future GPU machine."""
import torch


def get_device(prefer="auto"):
    """Return a torch device.

    Pass cpu to force the CPU, cuda to require a GPU when available, or auto to use the GPU
    when one is present and fall back to the CPU otherwise.
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
