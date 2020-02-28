import logging
import time

import wandb

logger = logging.getLogger(__name__)


class timeit:
    """Decorator to measure length of time spent in the block in millis and log
  it to TensorBoard."""

    def __init__(self, tag="", step: int = None, noop=False):
        self.tag = tag
        self.noop = noop
        self.step = step

    def __enter__(self):
        if self.noop:
            return self
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_args):
        if self.noop:
            return
        self.end = time.perf_counter()
        interval_ms = 1000 * (self.end - self.start)
        newtag = "times/" + self.tag
        wandb.log({newtag: interval_ms}, step=self.step)
