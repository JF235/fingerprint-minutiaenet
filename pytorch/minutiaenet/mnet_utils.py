from typing import Optional
import sys
import logging
import time
import os
import torch

DEFAULT_COARSENET_WEIGHTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../Models/CoarseNet.pth")
)
DEFAULT_FINENET_WEIGHTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../Models/FineNet.pth")
)
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_minutiaenet_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a logger that prefixes messages with '[MinutiaeNet]'."""
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        fmt = "[MinutiaeNet] %(levelname)s: %(message)s"
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(fmt))
        h.setLevel(logging.NOTSET)
        logger.addHandler(h)

    logger.propagate = False
    return logger


class MnetTimer:
    """Timer context manager with DEBUG-level logging."""
    def __init__(self, name: str, logger: logging.Logger):
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        self.logger.debug(f"{self.name} - {self.interval:.3f} s")
