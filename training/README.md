# Training CSRNet

## Quick Start

### 1. JSON data files

The training script needs two JSON files — one listing training image paths and
one listing test/validation image paths.  Pre-built JSON files for ShanghaiTech
Parts A and B are included in this folder:

| File | Part | Split | Images |
|------|------|-------|--------|
| `part_A_train.json` | A (dense) | train | 300 |
| `part_A_test.json`  | A (dense) | test  | 182 |
| `part_B_train.json` | B (sparse) | train | 400 |
| `part_B_test.json`  | B (sparse) | test  | 316 |

Each JSON file is a flat list of image paths:

```json
[
  "ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/images/IMG_1.jpg",
  "ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/images/IMG_2.jpg",
  ...
]
```

Ground truth (`.mat` files) is resolved automatically — `dataset.py` looks for
`GT_<stem>.mat` in a sibling `ground_truth/` folder next to `images/`.

### 2. Run training

From the **project root** directory:

```bash
# Train on Part A (dense crowds), GPU 0, save checkpoints with prefix "partA_"
python train.py training/part_A_train.json training/part_A_test.json 0 partA_ --epochs 400

# Train on Part B (sparse crowds), GPU 0
python train.py training/part_B_train.json training/part_B_test.json 0 partB_ --epochs 400
```

#### Argument order

All four **positional** arguments must come before any optional flags:

```
python train.py TRAIN TEST GPU TASK [--epochs N] [--pre PATH]
```

| Argument | Positional? | Description |
|----------|:-----------:|-------------|
| `train_json` | yes | Path to the training JSON file |
| `test_json` | yes | Path to the test/validation JSON file |
| `gpu` | yes | CUDA device id (e.g. `0`). Training falls back to CPU if no GPU is available |
| `task` | yes | Prefix for checkpoint filenames (e.g. `partA_` produces `partA_checkpoint.pth`) |
| `--epochs N` | no | Number of epochs (default 400) |
| `--pre PATH` | no | Resume from a previously saved checkpoint |

**Common mistake:** putting `--epochs` before `task`. This makes argparse think
`task` is missing. Always place the four positional args first:

```bash
# Correct
python train.py training/part_A_train.json training/part_A_test.json 0 partA_ --epochs 400

# Wrong — "task" is missing because argparse consumes "0" as GPU
python train.py training/part_A_train.json training/part_A_test.json 0 --epochs 400
```

### 3. Resume from a checkpoint

```bash
python train.py training/part_A_train.json training/part_A_test.json 0 partA_ --pre partA_checkpoint.pth
```

This restores the model weights, optimizer state, and epoch counter so training
continues exactly where it left off.

### 4. Output

Two files are saved after every epoch under the `task` prefix:

- `<task>checkpoint.pth` — latest checkpoint (model weights, optimizer state, epoch, best MAE).
- `<task>model_best.pth` — copy of whichever checkpoint achieved the lowest validation MAE so far.

Use `load_csrnet_model` from `csrnet.py` to load the best model for inference:

```python
from csrnet import load_csrnet_model

model = load_csrnet_model("partA_model_best.pth", map_location="cpu")
```

### 5. What happens during training

Each epoch:

1. **Learning rate** is adjusted via a piecewise schedule (see `adjust_learning_rate`).
2. **Training**: every image is forward-passed through CSRNet, MSE loss is computed
   between the predicted and target density maps, and weights are updated with SGD.
3. **Validation**: MAE (mean absolute count error) is computed over the test set.
4. **Checkpoint**: the current state is saved; if this epoch's MAE is the best yet,
   `model_best.pth` is updated.
