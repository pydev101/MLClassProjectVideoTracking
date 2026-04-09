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

Usage example:
    python train.py path/to/train.json path/to/val.json 0 ./runs/exp1_ --epochs 400

Positional args: train_json, test_json, gpu (CUDA device id), task (checkpoint prefix).
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import dataset
from csrnet import CSRNet
from utils import save_checkpoint

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu',metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task',metavar='TASK', type=str,
                    help='task id to use.')

parser.add_argument('--epochs', type=int, default=400,
                    help='number of epochs to train (default: 400)')

def main():
    global args,best_prec1
    
    best_prec1 = 1e6
    
    args = parser.parse_args()
    args.original_lr = 1e-7
    args.lr = 1e-7
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    # Piecewise LR schedule: at each epoch in `steps`, multiply LR by the
    # corresponding entry in `scales`.  The first entry (-1) means "from the
    # start", so the initial LR is used unchanged until epoch 1.
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30

    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)
    
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
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
            
    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch)
        
        train(train_list, model, criterion, optimizer, epoch, device)
        prec1 = validate(val_list, model, device)
        
        # Track the best (lowest) validation MAE across all epochs.
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '.format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,args.task)

def train(train_list, model, criterion, optimizer, epoch, device):
    """
    Run one training epoch over all images in ``train_list``.

    For each mini-batch (batch_size=1 by default):
      1. Load the image and its density target via ``dataset.listDataset``.
      2. Forward-pass through CSRNet to get a predicted density map.
      3. If the target and output spatial sizes disagree (odd image dimensions),
         bilinearly resize the target and rescale it so the count is preserved.
      4. Compute MSE loss, back-propagate, and update weights.
    """
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    
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
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    
    model.train()
    end = time.time()
    
    for i,(img, target)in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        img = img.to(device)
        output = model(img)

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
        
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))
    
def validate(val_list, model, device):
    """
    Evaluate model on the validation set and return the Mean Absolute Error
    (MAE) of the crowd-count prediction.

    MAE is computed as the average absolute difference between the *sum* of
    the predicted density map and the *sum* of the ground-truth density map
    across all validation images.  Lower is better.
    """
    print("begin test")
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
        
    mae = mae/len(test_loader)    
    print(' * MAE {mae:.3f} '
              .format(mae=mae))

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
