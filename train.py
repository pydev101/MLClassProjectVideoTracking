"""
CSRNet training script.

Trains a CSRNet crowd-density model using SGD with momentum on image/density
pairs loaded through ``dataset.listDataset``.  The training loop:

  1. Builds the model with VGG-16 frontend weights (ImageNet transfer).
  2. Optionally resumes from a saved checkpoint (``--pre``).
  3. For each epoch:
     a. Adjusts the learning rate via a piecewise schedule.
     b. Runs one pass over the training set (MSE loss on density maps).
     c. Evaluates MAE on the validation set.
     d. Saves a checkpoint (and copies it as "best" when MAE improves).

Usage examples:
    python train.py path/to/train.json path/to/val.json --gpu 0 --task ./runs/exp1_ --epochs 400
    python train.py --labeled-dir ./tools/labeling/labeled_data --gpu 0 --task ./runs/labeled_

If ``train_json`` is a **combined** file (JSON object with ``"train"`` and ``"test"``
lists), both splits are read from that file. Pass the same path again as
``test_json`` to satisfy the CLI, or any path (the second file is ignored when
the first is combined).

When ``--labeled-dir`` is given, the training/validation lists are built
automatically by scanning the labeling tool's output directory for source
images with matching JSON manifests.  No JSON file args are needed.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import dataset
from csrnet import CSRNet
import shutil

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from rich.align import Align
from rich.box import ROUNDED
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


parser = argparse.ArgumentParser(description='PyTorch CSRNet')
parser.add_argument('train_json', metavar='TRAIN', nargs='?', default=None, help='path to train json')
parser.add_argument('test_json', metavar='TEST', nargs='?', default=None, help='path to test json')
parser.add_argument(
    'legacy_gpu',
    nargs='?',
    default=None,
    metavar='GPU',
    help='GPU id (optional; same as --gpu, for CSRNet-style: TRAIN TEST GPU TASK)',
)
parser.add_argument(
    'legacy_task',
    nargs='?',
    default=None,
    metavar='TASK',
    help='checkpoint prefix (optional; same as --task)',
)
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str, help='path to the pretrained model')
parser.add_argument('--gpu', '-g', default=argparse.SUPPRESS, type=str, help='GPU id to use (default: 0)')
parser.add_argument(
    '--task',
    default=argparse.SUPPRESS,
    type=str,
    help='checkpoint prefix (default: labeled_)',
)
parser.add_argument('--epochs', type=int, default=400, help='number of epochs to train (default: 400)')
parser.add_argument('--labeled-dir', type=str, default=None,
    help='path to labeled_data directory from the labeling tool; overrides train/test json')
parser.add_argument('--val-fraction', type=float, default=0.2,
    help='fraction of labeled images held out for validation (default: 0.2)')


def _fmt_duration(seconds: float) -> str:
    if seconds < 0 or not math.isfinite(seconds):
        return "—"
    total = int(max(0, round(seconds)))
    h, r = divmod(total, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


class TrainingStatusLine:
    """Rich live status for throughput, epoch progress, elapsed time, and ETA."""

    def __init__(self, console: Console, args: argparse.Namespace) -> None:
        self.console = console
        self.args = args
        self.train_t0 = time.perf_counter()
        self.live: Optional[Live] = None
        self.phase = "warmup"
        self.epoch = 0
        self.batch_i = 0
        self.n_batches = 1
        self.loss_val = 0.0
        self.loss_avg = 0.0
        self.img_per_s = 0.0
        self.epoch_per_s = 0.0
        self.eta_sec = float("nan")
        self.val_i = 0
        self.val_n = 0
        self.total_epochs_span = max(1, args.epochs - args.start_epoch)

    def refresh(self) -> None:
        if self.live is not None:
            self.live.update(self.render())

    def render(self) -> Panel:
        elapsed = time.perf_counter() - self.train_t0
        sep = Text("  ", style="dim")
        row = Text()

        row.append("◆ ", style="bold magenta")
        row.append(f"epoch ", style="dim")
        row.append(
            f"{self.epoch + 1}/{self.args.epochs}",
            style="bold yellow",
        )
        row.append_text(sep)
        row.append("batch ", style="dim")
        pct = 100.0 * (self.batch_i + 1) / max(1, self.n_batches)
        row.append(
            f"{self.batch_i + 1}/{self.n_batches}",
            style="bold green",
        )
        row.append(f" ({pct:.0f}%)", style="green")

        row.append_text(sep)
        row.append("loss ", style="dim")
        row.append(f"{self.loss_val:.4f}", style="cyan")
        row.append(" (μ ", style="dim")
        row.append(f"{self.loss_avg:.4f}", style="bold cyan")
        row.append(")", style="dim")

        row.append_text(sep)
        row.append("lr ", style="dim")
        row.append(f"{self.args.lr:.2e}", style="white")

        row2 = Text()
        row2.append("⚡ ", style="bold bright_yellow")
        row2.append("imgs/s ", style="dim")
        row2.append(
            f"{self.img_per_s:.2f}",
            style="bold bright_cyan",
        )
        row2.append_text(sep)
        row2.append("epochs/s ", style="dim")
        row2.append(
            f"{self.epoch_per_s:.4f}",
            style="bold bright_green",
        )
        row2.append_text(sep)
        row2.append("⏱ elapsed ", style="dim")
        row2.append(_fmt_duration(elapsed), style="bold white")
        row2.append_text(sep)
        row2.append("ETA ", style="dim")
        row2.append(_fmt_duration(self.eta_sec), style="bold bright_blue")

        if self.phase == "val":
            row2.append_text(sep)
            row2.append("◎ val ", style="dim")
            row2.append(
                f"{self.val_i + 1}/{max(1, self.val_n)}",
                style="magenta",
            )

        subtitle = Text()
        subtitle.append(self.phase.upper(), style="bold dim")

        inner = Group(Align.left(row), Align.left(row2))
        return Panel(
            inner,
            title="[bold white on blue] CSRNet [/] [dim]training[/]",
            subtitle=subtitle,
            subtitle_align="left",
            box=ROUNDED,
            border_style="bright_blue",
            padding=(0, 1),
        )

    def set_train_batch(
        self,
        epoch: int,
        batch_i: int,
        n_batches: int,
        batch_time_s: float,
        loss_val: float,
        loss_avg: float,
    ) -> None:
        self.phase = "train"
        self.epoch = epoch
        self.batch_i = batch_i
        self.n_batches = max(1, n_batches)
        self.loss_val = loss_val
        self.loss_avg = loss_avg

        if batch_time_s > 1e-9:
            self.img_per_s = self.args.batch_size / batch_time_s

        elapsed = time.perf_counter() - self.train_t0
        progress_epochs = (
            (epoch - self.args.start_epoch)
            + (batch_i + 1) / self.n_batches
        )
        if elapsed > 1e-6 and progress_epochs > 1e-9:
            self.epoch_per_s = progress_epochs / elapsed
        else:
            self.epoch_per_s = 0.0

        remaining = self.total_epochs_span - progress_epochs
        if progress_epochs > 1e-6 and remaining >= 0 and math.isfinite(elapsed):
            self.eta_sec = (elapsed / progress_epochs) * remaining
        else:
            self.eta_sec = float("nan")

        self.refresh()

    def start_validation(self, n_batches: int) -> None:
        self.phase = "val"
        self.val_i = 0
        self.val_n = max(1, n_batches)
        self.refresh()

    def validation_step(self, batch_i: int) -> None:
        self.val_i = batch_i
        self.refresh()

    def end_validation(self) -> None:
        self.phase = "train"

    def start_epoch_message(self, epoch: int, n_batches: int) -> None:
        self.epoch = epoch
        self.n_batches = max(1, n_batches)
        self.batch_i = -1
        self.phase = "train"


class TrainingPlotter:
    """Renders and saves loss / MAE curves during training.

    After each epoch call ``update()`` to append the latest metrics and
    overwrite the live PNG.  Call ``save_final()`` after training to write an
    auto-incremented copy that won't be overwritten on the next run.
    """

    FIGURES_DIR = Path(__file__).resolve().parent / "figures" / "training"

    def __init__(self, task: str = "", save_dir: Optional[Path] = None) -> None:
        self.save_dir = save_dir or self.FIGURES_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.task = task.rstrip("_")
        self.epochs: List[int] = []
        self.losses: List[float] = []
        self.maes: List[float] = []
        label = f"training_curves_{self.task}" if self.task else "training_curves"
        self._live_path = self.save_dir / f"{label}.png"

    def update(self, epoch: int, loss: float, mae: float) -> None:
        self.epochs.append(epoch + 1)
        self.losses.append(loss)
        self.maes.append(mae)
        self._save_plot(self._live_path)

    def save_final(self) -> Path:
        """Save a copy with an auto-incremented filename."""
        label = f"training_curves_{self.task}_final" if self.task else "training_curves_final"
        path = self._next_available(label)
        self._save_plot(path)
        return path

    # ------------------------------------------------------------------

    def _next_available(self, base: str, ext: str = ".png") -> Path:
        candidate = self.save_dir / f"{base}{ext}"
        counter = 1
        while candidate.exists():
            candidate = self.save_dir / f"{base}_{counter}{ext}"
            counter += 1
        return candidate

    def _save_plot(self, path: Path) -> None:
        fig = Figure(figsize=(10, 8))
        FigureCanvasAgg(fig)
        ax_loss, ax_mae = fig.subplots(2, 1, sharex=True)

        ax_loss.plot(self.epochs, self.losses, "b-o", markersize=2, linewidth=1, label="Training Loss")
        ax_loss.set_ylabel("Loss (MSE)")
        title = f"Training Curves — {self.task}" if self.task else "Training Curves"
        ax_loss.set_title(title)
        ax_loss.legend(loc="upper right")
        ax_loss.grid(True, alpha=0.3)

        ax_mae.plot(self.epochs, self.maes, "r-o", markersize=2, linewidth=1, label="Validation MAE")
        ax_mae.set_ylabel("MAE")
        ax_mae.set_xlabel("Epoch")
        ax_mae.legend(loc="upper right")
        ax_mae.grid(True, alpha=0.3)

        if self.maes:
            best_val = min(self.maes)
            best_idx = self.maes.index(best_val)
            ax_mae.axhline(best_val, color="r", linestyle="--", alpha=0.4)
            ax_mae.annotate(
                f"Best: {best_val:.3f} (epoch {self.epochs[best_idx]})",
                xy=(self.epochs[best_idx], best_val),
                xytext=(10, 12),
                textcoords="offset points",
                fontsize=8,
                color="red",
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.5),
            )

        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")


def main():
    global args, best_prec1
    
    best_prec1 = 1e6
    args = parser.parse_args()
    if not hasattr(args, 'gpu'):
        args.gpu = args.legacy_gpu if args.legacy_gpu is not None else '0'
    if not hasattr(args, 'task'):
        args.task = args.legacy_task if args.legacy_task is not None else 'labeled_'
    args.original_lr = 1e-7
    args.lr = 1e-7
    # Process one image per iteration.  Crowd images have varying dimensions
    # and can't easily be stacked into a uniform tensor, so batch_size=1 is
    # standard for CSRNet (true stochastic gradient descent).
    args.batch_size    = 1
    # SGD momentum: 95 % of the previous gradient direction carries into the
    # current update, smoothing out noisy per-image gradients and accelerating
    # convergence through flat regions of the loss landscape.
    args.momentum      = 0.95
    # Weight decay (L2 regularisation): penalises large weights to reduce
    # overfitting.  Each update effectively shrinks every weight by
    # (1 - lr * decay) before applying the gradient.
    args.decay         = 5*1e-4
    # Epoch to start from; overwritten when resuming from a checkpoint.
    args.start_epoch   = 0

    # Piecewise LR schedule: at each epoch in `steps`, multiply LR by the
    # corresponding entry in `scales`.  The first entry (-1) means "from the
    # start", so the initial LR is used unchanged until epoch 1.
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30

    if args.labeled_dir:
        train_list, val_list = scan_labeled_dir(args.labeled_dir, args.val_fraction)
        print(f"=> Labeled dir: {args.labeled_dir}")
        print(f"   Training images: {len(train_list)}, Validation images: {len(val_list)}")
    elif args.train_json:
        with open(args.train_json, "r", encoding="utf-8") as outfile:
            train_blob = json.load(outfile)
        if isinstance(train_blob, dict) and "train" in train_blob and "test" in train_blob:
            train_list = train_blob["train"]
            val_list = train_blob["test"]
            p_train = os.path.normcase(os.path.abspath(args.train_json))
            p_test = os.path.normcase(os.path.abspath(args.test_json))
            if p_train != p_test:
                print(
                    "=> train_json is a combined file (train+test keys); ignoring TEST path:",
                    args.test_json,
                )
        else:
            train_list = train_blob
            with open(args.test_json, "r", encoding="utf-8") as outfile:
                val_list = json.load(outfile)
    else:
        parser.error("Provide train_json/test_json or --labeled-dir")
    
    # Pin the GPU visible to this process and seed all RNGs for reproducibility.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(int(args.seed) % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed) % (2**32))
    

    # load_weights=False triggers VGG-16 ImageNet weight transfer into the
    # frontend (see CSRNet.__init__).
    model = CSRNet(load_weights=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # MSE with reduction="sum" so the loss magnitude scales with image size,
    # which is the convention from the original CSRNet-pytorch repo.
    criterion = nn.MSELoss(reduction="sum").to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    # Optionally resume training from a previously saved checkpoint.
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            try:
                checkpoint = torch.load(args.pre, map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(args.pre, map_location=device)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # load weight and biases
            model.load_state_dict(checkpoint['state_dict'])
            # load optimizer state (gradients and whatnot)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
        

    console = Console()
    status = TrainingStatusLine(console, args)
    plotter = TrainingPlotter(task=args.task)
    live_ctx = Live(
        status.render(),
        console=console,
        refresh_per_second=12,
        transient=False,
    )
    status.live = live_ctx

    with live_ctx:
        # Train the model for the specified number of epochs
        for epoch in range(args.start_epoch, args.epochs):

            # Adjust the learning rate based on the epoch
            adjust_learning_rate(optimizer, epoch)
            epoch_loss = train(train_list, model, criterion, optimizer, epoch, device, status)
            prec1 = validate(val_list, model, device, status)

            plotter.update(epoch, epoch_loss, prec1)

            # Track the best (lowest) validation MAE across all epochs.
            is_best = prec1 < best_prec1
            best_prec1 = min(prec1, best_prec1)
            console.print(
                f"[bold green] * best MAE[/] [bold white]{best_prec1:.3f}[/]"
            )
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.pre,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.task)

    final_path = plotter.save_final()
    console.print(f"[bold green]Training curves saved:[/] {final_path}")

def scan_labeled_dir(
        labeled_dir: str, val_fraction: float = 0.2
    ) -> Tuple[List[str], List[str]]:
    """Scan a labeling-tool output directory and return ``(train_list, val_list)``.

    Discovers source images under ``<labeled_dir>/source/`` that have a matching
    JSON manifest in ``<labeled_dir>/ground_truth/json/``.  When fewer than 5
    images are available, all images are used for both training *and* validation
    so the model can at least overfit as a sanity check.
    """
    root = Path(labeled_dir)
    source_dir = root / "source"
    manifest_dir = root / "ground_truth" / "json"

    if not source_dir.is_dir():
        raise FileNotFoundError(f"No source/ subdirectory in {labeled_dir}")

    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
    images: List[str] = []
    for img_path in sorted(source_dir.iterdir()):
        if img_path.is_file() and img_path.suffix.lower() in IMG_EXT:
            manifest = manifest_dir / f"{img_path.stem}.json"
            if manifest.is_file():
                images.append(str(img_path))

    if not images:
        raise FileNotFoundError(
            f"No labeled images with matching manifests found in {labeled_dir}"
        )

    if len(images) < 5:
        return list(images), list(images)

    random.shuffle(images)
    n_val = max(1, int(len(images) * val_fraction))
    return images[n_val:], images[:n_val]


def train(
    train_list,
    model,
    criterion,
    optimizer,
    epoch,
    device,
    status: Optional[TrainingStatusLine],
) -> float:
    """
    Run one training epoch over all images in ``train_list``.

    For each mini-batch (batch_size=1 by default):
      1. Load the image and its density target via ``dataset.listDataset``.
      2. Forward-pass through CSRNet to get a predicted density map.
      3. If the target and output spatial sizes disagree (odd image dimensions),
         bilinearly resize the target and rescale it so the count is preserved.
      4. Compute MSE loss, back-propagate, and update weights.

    Returns:
        Average training loss for this epoch.
    """
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # Create a data loader for the training set
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
            shuffle=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            ]), 
            train=True, 
            seen=model.seen,
            batch_size=args.batch_size,
            num_workers=args.workers),
            batch_size=args.batch_size
        )
    n_batches = len(train_loader)
    if status is not None:
        status.start_epoch_message(epoch, n_batches)
        status.refresh()
    else:
        print(
            "epoch %d, processed %d samples, lr %.10f"
            % (epoch, epoch * len(train_loader.dataset), args.lr)
        )

    # Set the model to training mode
    model.train()
    end = time.time()
    
    # for each image in the training set, 
    for i,(img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        # Move the image and target to the device
        img = img.to(device)
        # Run the model forward pass
        output = model(img)
        # Move the target to the device
        target = target.to(device)

        # Ensure target is 4-D [N, 1, H, W] to match network output.
        if target.dim() == 3:
            target = target.unsqueeze(0)
        # When the target spatial size doesn't exactly match the output (can
        # happen with odd-sized images after 3 max-pools), resize the target
        # and rescale so the density sum (crowd count) is preserved.
        if output.shape != target.shape:
            tsum = target.sum()
            target = F.interpolate(
                target, size=output.shape[2:], mode="bilinear", align_corners=False
            )
            target = target * (tsum / (target.sum() + 1e-8))

        loss = criterion(output, target)
        
        # back propagation and update weights
        losses.update(loss.item(), img.size(0))
        # zero the gradients
        optimizer.zero_grad()
        # back propagate the loss
        loss.backward()
        # update weights
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()

        if status is not None:
            status.set_train_batch(
                epoch,
                i,
                n_batches,
                batch_time.avg,
                losses.val,
                losses.avg,
            )
        elif i % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    epoch,
                    i,
                    n_batches,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                )
            )

    return losses.avg

def validate(
    val_list,
    model,
    device,
    status: Optional[TrainingStatusLine],
):
    """
    Evaluate model on the validation set and return the Mean Absolute Error
    (MAE) of the crowd-count prediction.

    MAE is computed as the average absolute difference between the *sum* of
    the predicted density map and the *sum* of the ground-truth density map
    across all validation images.  Lower is better.
    """
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                    shuffle=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                    ]),  train=False),
    batch_size=args.batch_size)

    model.eval()

    mae = 0.0
    if len(test_loader) == 0:
        return mae

    if status is not None:
        status.start_validation(len(test_loader))

    for i, (img, target) in enumerate(test_loader):
        img = img.to(device)
        output = model(img)
        target = target.to(device)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        # Same spatial-mismatch handling as in the training loop.
        if output.shape != target.shape:
            tsum = target.sum()
            target = F.interpolate(
                target, size=output.shape[2:], mode="bilinear", align_corners=False
            )
            target = target * (tsum / (target.sum() + 1e-8))
        # MAE is based on the *count* (sum of density), not per-pixel error.
        mae += torch.abs(output.sum() - target.sum()).item()
        if status is not None:
            status.validation_step(i)

    mae = mae / len(test_loader)
    if status is not None:
        status.end_validation()
        status.console.print(f"[dim] * MAE[/] [bold]{mae:.3f}[/]")
    else:
        print(" * MAE {mae:.3f} ".format(mae=mae))

    return mae
        
def adjust_learning_rate(optimizer, epoch):
    """Apply the piecewise learning-rate schedule defined by ``args.steps``
    and ``args.scales``.

    Starting from ``original_lr``, the LR is multiplied by each scale whose
    corresponding step has been reached.  In the default config all scales are
    1, so the LR stays constant.  Override ``steps`` and ``scales`` for a
    step-decay schedule (e.g. divide by 10 every 30 epochs).
    """
    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        
        scale = args.scales[i] if i < len(args.scales) else 1
        
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

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

        
class AverageMeter(object):
    """Tracks a running average of a scalar value (e.g. loss, timing).

    Call ``update(val)`` after each batch; read ``.avg`` for the mean so far.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
if __name__ == '__main__':
    main()