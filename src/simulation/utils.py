# Utility helpers for the simulation package
import os
from typing import Iterable

# ---------------------------------------------------------------------------
# Configuration flags
# ---------------------------------------------------------------------------
# FAST_MODE can be toggled via the environment variable ``FAST_MODE``.
# When enabled we use larger batch sizes, more workers and enable AMP
# (automatic mixed precision) where appropriate.  The default is ``False``
# which keeps the training lightweight and suitable for CI environments.
FAST_MODE: bool = os.getenv("FAST_MODE", "0") in {"1", "true", "True"}

# ---------------------------------------------------------------------------
# Progress bar helper
# ---------------------------------------------------------------------------
def _make_progress_bar(current: int, total: int, bar_length: int = 20) -> str:
    """Return a simple textual progress bar.

    Args:
        current: Current step (1‑based).
        total: Total number of steps.
        bar_length: Length of the bar in characters.

    Returns:
        A string like ``[██████......]`` where the filled portion reflects the
        progress.  The function is deliberately lightweight and has no external
        dependencies so it can be used in any part of the simulation code.
    """
    if total <= 0:
        raise ValueError("total must be a positive integer")
    # Clamp current to the range [0, total]
    current = max(0, min(current, total))
    filled = int(bar_length * current / total)
    bar = "\u2588" * filled + "\u2591" * (bar_length - filled)
    return f"[{bar}]"

# ---------------------------------------------------------------------------
# Miscellaneous helpers (placeholder for future extensions)
# ---------------------------------------------------------------------------
def chunk_iterable(iterable: Iterable, chunk_size: int):
    """Yield successive ``chunk_size`` sized chunks from *iterable*.

    This helper is useful when we need to split a list of file paths or
    dataset indices into smaller batches for processing.
    """
    it = iter(iterable)
    while True:
        chunk = list()
        try:
            for _ in range(chunk_size):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk

__all__ = ["FAST_MODE", "_make_progress_bar", "chunk_iterable"]
