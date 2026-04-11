"""
Run CSRNet crowd-density inference on a video and write output videos with
heatmap overlays.

Produces up to three output files (controlled via CLI flags):
  - *_overlay.mp4  : original frames with the density heatmap blended on top
  - *_heatmap.mp4  : standalone density heatmap (no base image)
  - *_sidebyside.mp4 : original frame next to the heatmap, side by side

Usage examples
--------------
# Overlay only (default):
  python test_video.py --video input.mp4 --weights all_model_best.pth

# All three outputs:
  python test_video.py --video input.mp4 --weights all_model_best.pth --overlay --heatmap --sidebyside

# Use GPU if available:
  python test_video.py --video input.mp4 --weights all_model_best.pth --device cuda
"""

import argparse
import sys
import time
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm as CM

from csrnet import load_csrnet_model

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def density_to_colormap(density: np.ndarray, cmap=CM.jet) -> np.ndarray:
    """Convert a 2-D density map to an RGB uint8 image via a matplotlib colormap.

    The density is min-max normalised so the full colour range is used per frame.
    Returns an ``(H, W, 3)`` BGR array ready for ``cv2.imwrite`` / ``VideoWriter``.
    """
    d = density.astype(np.float64)
    lo, hi = d.min(), d.max()
    if hi - lo > 1e-8:
        d = (d - lo) / (hi - lo)
    else:
        d = np.zeros_like(d)
    rgba = cmap(d)  # (H, W, 4) float 0-1
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def overlay_heatmap(
    frame_bgr: np.ndarray,
    density: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Alpha-blend a colormapped density map onto a BGR video frame."""
    heat_bgr = density_to_colormap(density)
    heat_bgr = cv2.resize(heat_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]))
    blended = cv2.addWeighted(frame_bgr, 1 - alpha, heat_bgr, alpha, 0)
    return blended


def burn_count_text(frame: np.ndarray, count: float) -> np.ndarray:
    """Draw the estimated crowd count in the top-left corner."""
    text = f"Count: {count:.1f}"
    cv2.putText(
        frame, text, (10, 36),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, (10, 36),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return frame


def make_writer(
    path: Path, fourcc: int, fps: float, width: int, height: int,
) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    return writer


def _read_batches(cap: cv2.VideoCapture, batch_size: int, q: "Queue[Optional[List[np.ndarray]]]") -> None:
    """Background thread: read frames and enqueue them in batches."""
    while True:
        frames: List[np.ndarray] = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        if frames:
            q.put(frames)
        if len(frames) < batch_size:
            q.put(None)
            break


def _frames_to_batch(frames_bgr: List[np.ndarray], device: torch.device) -> torch.Tensor:
    """Convert BGR uint8 frames directly to a normalised (N,3,H,W) tensor."""
    tensors = []
    for f in frames_bgr:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
        tensors.append(t)
    batch = torch.stack(tensors)
    batch.sub_(_MEAN).div_(_STD)
    return batch.to(device, non_blocking=True)


def _upscale_densities(density_batch: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Bilinear-upscale a (N,1,h',w') density batch to (N,1,H,W), preserving counts."""
    sums = density_batch.flatten(1).sum(1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
    up = F.interpolate(density_batch, size=(h, w), mode="bilinear", align_corners=False)
    up_sums = up.flatten(1).sum(1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
    return up * (sums / (up_sums + 1e-8))


def process_video(
    video_path: str,
    weights_path: str,
    *,
    device: str = "cpu",
    alpha: float = 0.45,
    write_overlay: bool = True,
    write_heatmap: bool = False,
    write_sidebyside: bool = False,
    output_dir: Optional[str] = None,
    batch_size: int = 8,
) -> None:
    video = Path(video_path)
    if not video.is_file():
        print(f"Error: video not found: {video}")
        sys.exit(1)

    out_dir = Path(output_dir) if output_dir else video.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem

    model = load_csrnet_model(weights_path, map_location=device)
    dev = next(model.parameters()).device

    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True

    use_amp = dev.type == "cuda"

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"Error: cannot open video {video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writers: dict[str, cv2.VideoWriter] = {}
    if write_overlay:
        writers["overlay"] = make_writer(
            out_dir / f"{stem}_overlay.mp4", fourcc, fps, w, h,
        )
    if write_heatmap:
        writers["heatmap"] = make_writer(
            out_dir / f"{stem}_heatmap.mp4", fourcc, fps, w, h,
        )
    if write_sidebyside:
        writers["sidebyside"] = make_writer(
            out_dir / f"{stem}_sidebyside.mp4", fourcc, fps, w * 2, h,
        )

    if not writers:
        print("No output type selected. Use --overlay, --heatmap, or --sidebyside.")
        cap.release()
        sys.exit(1)

    frame_idx = 0
    t0 = time.time()
    print(f"Processing {video.name}  ({w}x{h} @ {fps:.1f} fps, {total_frames} frames)")
    print(f"Device: {dev}  |  batch_size: {batch_size}  |  AMP: {use_amp}")
    print(f"Outputs: {', '.join(writers.keys())}")

    q: Queue[Optional[List[np.ndarray]]] = Queue(maxsize=3)
    reader = Thread(target=_read_batches, args=(cap, batch_size, q), daemon=True)
    reader.start()

    try:
        while True:
            frames_bgr = q.get()
            if frames_bgr is None:
                break

            batch_tensor = _frames_to_batch(frames_bgr, dev)

            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = model(batch_tensor)
                    outputs = outputs.float()
                else:
                    outputs = model(batch_tensor)

            densities_full = _upscale_densities(outputs, h, w)
            densities_np = densities_full.squeeze(1).cpu().numpy()
            counts = outputs.flatten(1).sum(1).cpu().numpy()

            for i, frame_bgr in enumerate(frames_bgr):
                density_full = densities_np[i]
                count = float(counts[i])

                if "overlay" in writers:
                    out = overlay_heatmap(frame_bgr, density_full, alpha)
                    burn_count_text(out, count)
                    writers["overlay"].write(out)

                if "heatmap" in writers:
                    heat = density_to_colormap(density_full)
                    heat = cv2.resize(heat, (w, h))
                    burn_count_text(heat, count)
                    writers["heatmap"].write(heat)

                if "sidebyside" in writers:
                    heat = density_to_colormap(density_full)
                    heat = cv2.resize(heat, (w, h))
                    burn_count_text(heat, count)
                    orig = frame_bgr.copy()
                    burn_count_text(orig, count)
                    sbs = np.hstack([orig, heat])
                    writers["sidebyside"].write(sbs)

                frame_idx += 1
                if frame_idx % 50 == 0 or frame_idx == total_frames:
                    elapsed = time.time() - t0
                    fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                    pct = frame_idx / total_frames * 100 if total_frames else 0
                    print(
                        f"  frame {frame_idx}/{total_frames}"
                        f"  ({pct:5.1f}%)  {fps_actual:.1f} fps",
                        flush=True,
                    )
    finally:
        reader.join(timeout=2)
        cap.release()
        for wr in writers.values():
            wr.release()

    elapsed = time.time() - t0
    print(f"\nDone — {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")
    for tag in writers:
        print(f"  {out_dir / f'{stem}_{tag}.mp4'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CSRNet on a video and produce density-heatmap output videos.",
    )
    parser.add_argument("--video", required=True, help="Path to input video file.")
    parser.add_argument(
        "--weights", required=True,
        help="Path to CSRNet .pth checkpoint.",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device (default: cpu). Use 'cuda' for GPU.",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity (0-1).")
    parser.add_argument("--batch-size", type=int, default=8, help="Frames per inference batch (default: 8).")
    parser.add_argument("--output-dir", default=None, help="Directory for output videos (default: same as input).")

    mode = parser.add_argument_group("output modes (at least one required)")
    mode.add_argument("--overlay", action="store_true", help="Write overlay video (original + heatmap).")
    mode.add_argument("--heatmap", action="store_true", help="Write standalone heatmap video.")
    mode.add_argument("--sidebyside", action="store_true", help="Write side-by-side video (original | heatmap).")

    args = parser.parse_args()

    if not (args.overlay or args.heatmap or args.sidebyside):
        args.overlay = True

    process_video(
        args.video,
        args.weights,
        device=args.device,
        alpha=args.alpha,
        write_overlay=args.overlay,
        write_heatmap=args.heatmap,
        write_sidebyside=args.sidebyside,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
