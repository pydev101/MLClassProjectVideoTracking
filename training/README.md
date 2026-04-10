# Training CSRNet

All commands run from the **project root**.

## Datasets

Pre-built JSON files in this folder point to images from ShanghaiTech (Parts A & B) and the Mall dataset.

| File | Dataset | Split | Images |
|------|---------|-------|-------:|
| `part_A_train.json` | ShanghaiTech A (dense) | train | 300 |
| `part_A_test.json` | ShanghaiTech A (dense) | test | 182 |
| `part_B_train.json` | ShanghaiTech B (sparse) | train | 400 |
| `part_B_test.json` | ShanghaiTech B (sparse) | test | 316 |
| `mall_train.json` | Mall | train | 1,300 |
| `mall_test.json` | Mall | test | 700 |
| `combined.json` | A + B + Mall | both | 2,000 train / 1,198 test |

**Single-dataset files** are plain JSON arrays of image paths. **`combined.json`** is a JSON object with `"train"` and `"test"` keys — pass it as both positional arguments (the second is ignored).

## Train

```bash
# All datasets combined
python train.py training/combined.json training/combined.json 0 all_ --epochs 400

# ShanghaiTech Part A only
python train.py training/part_A_train.json training/part_A_test.json 0 partA_ --epochs 400

# ShanghaiTech Part B only
python train.py training/part_B_train.json training/part_B_test.json 0 partB_ --epochs 400

# Mall only
python train.py training/mall_train.json training/mall_test.json 0 mall_ --epochs 400
```


| Argument | Description |
|----------|-------------|
| 1st path | Training JSON (or combined JSON with `train`/`test` keys) |
| 2nd path | Validation JSON (ignored when the first file is combined) |
| `0` | GPU id (`0`, `1`, … - falls back to CPU if unavailable) |
| `all_` | Checkpoint prefix - any string you choose. Determines output filenames (e.g. `all_` → `all_checkpoint.pth`, `all_model_best.pth`). Use whatever makes sense: `all_`, `partA_`, `myrun_`, `runs/exp1_`, etc. |
| `--epochs N` | Total epochs to run (default 400) |

## How training works

The model is [CSRNet](https://arxiv.org/abs/1712.01140) — a VGG-16 frontend (pretrained on ImageNet) followed by dilated-convolution backend layers that output a density map. The sum of the density map gives the predicted crowd count.

Each epoch:

1. **Train** — every image is passed through the network one at a time (`batch_size=1`, since crowd images have varying dimensions). MSE loss is computed between the predicted and ground-truth density maps, then back-propagated with SGD.
2. **Validate** — Mean Absolute Error (MAE) of the predicted count vs. ground-truth count is computed over the validation set. Lower is better.
3. **Checkpoint** — the model is saved. If this epoch's MAE is the best so far, `model_best.pth` is also updated.

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 1 | Required — images have different spatial dimensions and can't be batched |
| Optimizer | SGD | With momentum |
| Learning rate | 1e-7 | Constant by default (piecewise schedule with all-1 scales) |
| Momentum | 0.95 | Smooths noisy single-image gradients |
| Weight decay | 5e-4 | L2 regularization to reduce overfitting |
| LR schedule | `[-1, 1, 100, 150]` × `[1, 1, 1, 1]` | Multiply LR by the scale at each step epoch. All 1s = no decay. Change `scales` in `train.py` for step-decay (e.g. `[1, 1, 0.1, 0.1]` to drop LR at epochs 100 & 150) |
| Loss | MSE (sum reduction) | Loss scales with image size, matching the original CSRNet convention |

These are hardcoded near the top of `main()` in `train.py`. Edit them there to experiment.

## Resume Training (Finetune)

```bash
python train.py training/combined.json training/combined.json 0 all_ --epochs 400 --pre all_checkpoint.pth
```

`--pre` loads weights, optimizer state, and epoch counter so training picks up where it left off. To train **longer**, increase `--epochs` beyond the saved epoch.

## Output

Each epoch saves two files under the checkpoint prefix:

- **`<prefix>checkpoint.pth`** - latest state (use with `--pre` to resume).
- **`<prefix>model_best.pth`** - copy of whichever epoch had the lowest validation MAE.
