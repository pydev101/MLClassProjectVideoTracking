"""
Training utilities (checkpoint I/O).

Provides ``save_checkpoint`` which persists the full training state (model
weights, optimizer state, epoch counter, best MAE) to disk after every epoch.
When the current epoch achieves the best validation MAE so far, the checkpoint
is also copied to a ``model_best.pth`` file so the best weights are always
accessible without scanning all epochs.
"""

import os
import shutil
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], is_best: bool, task_id: str, filename: str = "checkpoint.pth") -> None:
    """Save training checkpoint; copy to ``model_best.pth`` when ``is_best``.

    Files are written to ``<task_id><filename>`` (e.g. ``./runs/exp1_checkpoint.pth``).
    The ``task_id`` prefix lets you keep multiple experiments side-by-side in
    the same directory.
    """
    path = task_id + filename
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(state, path, _use_new_zipfile_serialization=True)
    if is_best:
        best = task_id + "model_best.pth"
        shutil.copyfile(path, best)
