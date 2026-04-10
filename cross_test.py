"""
Cross-dataset validation for CSRNet models.

Evaluates one or more trained CSRNet checkpoints against the ShanghaiTech
Part A, Part B, and Mall test sets, printing a comparison table of MAE and
MSE metrics.  Useful for measuring how well a model generalises across crowd
density regimes (Part A = dense, Part B = sparse, Mall = surveillance).

Usage:
    # Auto-discover all .pth files in the repo:
    python cross_test.py

    # Specify models explicitly:
    python cross_test.py --models all_100_model_best.pth models/csr_net_base/PartAmodel_best.pth

    # Use a GPU:
    python cross_test.py --gpu 0
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

import dataset
from csrnet import CSRNet

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text


REPO_ROOT = Path(__file__).resolve().parent
PART_A_TEST = REPO_ROOT / "training" / "part_A_test.json"
PART_B_TEST = REPO_ROOT / "training" / "part_B_test.json"
MALL_TEST = REPO_ROOT / "training" / "mall_test.json"

KNOWN_MODEL_LOCATIONS = [
    REPO_ROOT,
    REPO_ROOT / "models" / "csr_net_base",
]

console = Console()


def discover_models() -> List[Path]:
    """Find all .pth model files in known locations."""
    found: List[Path] = []
    for loc in KNOWN_MODEL_LOCATIONS:
        if loc.is_dir():
            found.extend(sorted(loc.glob("*model_best*.pth")))
    return found


def load_model(weights_path: Path, device: torch.device) -> CSRNet:
    model = CSRNet(load_weights=True)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


def evaluate(
    model: CSRNet,
    image_list: List[str],
    device: torch.device,
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None,
) -> Dict[str, float]:
    """Run inference on every image and return MAE, MSE, and mean GT count."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    loader = torch.utils.data.DataLoader(
        dataset.listDataset(image_list, shuffle=False, transform=transform, train=False),
        batch_size=1,
    )

    abs_errors: List[float] = []
    sq_errors: List[float] = []
    gt_counts: List[float] = []
    pred_counts: List[float] = []

    with torch.no_grad():
        for i, (img, target) in enumerate(loader):
            img = img.to(device)
            output = model(img)
            target = target.to(device)

            if target.dim() == 3:
                target = target.unsqueeze(0)
            if output.shape != target.shape:
                tsum = target.sum()
                target = F.interpolate(
                    target, size=output.shape[2:], mode="bilinear", align_corners=False
                )
                target = target * (tsum / (target.sum() + 1e-8))

            pred = output.sum().item()
            gt = target.sum().item()
            pred_counts.append(pred)
            gt_counts.append(gt)
            abs_errors.append(abs(pred - gt))
            sq_errors.append((pred - gt) ** 2)

            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1)

    mae = float(np.mean(abs_errors)) if abs_errors else 0.0
    mse = float(np.sqrt(np.mean(sq_errors))) if sq_errors else 0.0
    mean_gt = float(np.mean(gt_counts)) if gt_counts else 0.0
    mean_pred = float(np.mean(pred_counts)) if pred_counts else 0.0

    return {
        "mae": mae,
        "mse": mse,
        "mean_gt": mean_gt,
        "mean_pred": mean_pred,
        "n_images": len(abs_errors),
    }


def friendly_name(path: Path) -> str:
    """Shorten a model path to a readable label."""
    name = path.stem
    if path.parent != REPO_ROOT:
        name = f"{path.parent.name}/{name}"
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset CSRNet validation")
    parser.add_argument(
        "--models", nargs="+", type=str, default=None,
        help="Paths to .pth model checkpoints (default: auto-discover)",
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU id (default: 0)")
    parser.add_argument(
        "--part-a", type=str, default=str(PART_A_TEST),
        help="Path to Part A test JSON",
    )
    parser.add_argument(
        "--part-b", type=str, default=str(PART_B_TEST),
        help="Path to Part B test JSON",
    )
    parser.add_argument(
        "--mall", type=str, default=str(MALL_TEST),
        help="Path to Mall test JSON",
    )
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[dim]Device:[/] [bold]{device}[/]")

    if args.models:
        model_paths = [Path(m) for m in args.models]
    else:
        model_paths = discover_models()

    if not model_paths:
        console.print("[bold red]No models found.[/] Pass --models or place *model_best*.pth files in the repo root.")
        return

    console.print()
    console.print("[bold]Models to evaluate:[/]")
    for p in model_paths:
        console.print(f"  [cyan]{p}[/]")

    datasets: Dict[str, Tuple[str, List[str]]] = {}
    for label, json_path in [("Part A", args.part_a), ("Part B", args.part_b), ("Mall", args.mall)]:
        p = Path(json_path)
        if not p.exists():
            console.print(f"[yellow]Warning: {p} not found, skipping {label}[/]")
            continue
        with open(p, "r", encoding="utf-8") as f:
            img_list = json.load(f)
        if isinstance(img_list, dict):
            img_list = img_list.get("test", img_list.get("val", []))
        datasets[label] = (str(p), img_list)
        console.print(f"  [green]{label}[/]: {len(img_list)} images")

    if not datasets:
        console.print("[bold red]No test datasets found.[/]")
        return

    # --- Run evaluations ---
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    total_evals = len(model_paths) * sum(len(v[1]) for v in datasets.values())

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for model_path in model_paths:
            mname = friendly_name(model_path)
            console.print()
            console.print(f"[bold blue]Loading model:[/] {mname}")
            try:
                model = load_model(model_path, device)
            except Exception as e:
                console.print(f"  [bold red]Failed to load:[/] {e}")
                continue

            results[mname] = {}
            for ds_label, (ds_path, img_list) in datasets.items():
                task_id = progress.add_task(
                    f"  {mname} → {ds_label}", total=len(img_list)
                )
                metrics = evaluate(model, img_list, device, progress, task_id)
                results[mname][ds_label] = metrics
                progress.update(task_id, description=f"  [green]✓[/] {mname} → {ds_label}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Print results table ---
    console.print()

    table = Table(
        title="Cross-Dataset Validation Results",
        show_header=True,
        header_style="bold white on blue",
        border_style="bright_blue",
        title_style="bold white",
    )
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Dataset", style="green")
    table.add_column("Images", justify="right")
    table.add_column("MAE", justify="right", style="bold yellow")
    table.add_column("MSE (RMSE)", justify="right", style="bold red")
    table.add_column("Mean GT Count", justify="right", style="dim")
    table.add_column("Mean Pred Count", justify="right", style="dim")

    for mname, ds_results in results.items():
        first = True
        for ds_label, metrics in ds_results.items():
            table.add_row(
                mname if first else "",
                ds_label,
                str(metrics["n_images"]),
                f"{metrics['mae']:.2f}",
                f"{metrics['mse']:.2f}",
                f"{metrics['mean_gt']:.1f}",
                f"{metrics['mean_pred']:.1f}",
            )
            first = False
        if len(ds_results) > 0:
            table.add_section()

    console.print(table)

    # --- Per-model summary ---
    console.print()
    for mname, ds_results in results.items():
        if len(ds_results) < 2:
            continue
        labels = list(ds_results.keys())
        best_ds = min(labels, key=lambda l: ds_results[l]["mae"])
        worst_ds = max(labels, key=lambda l: ds_results[l]["mae"])
        console.print(
            f"[dim]{mname}:[/] best on [green]{best_ds}[/] "
            f"(MAE [bold]{ds_results[best_ds]['mae']:.2f}[/]), "
            f"worst on [red]{worst_ds}[/] "
            f"(MAE [bold]{ds_results[worst_ds]['mae']:.2f}[/])"
        )


if __name__ == "__main__":
    main()
