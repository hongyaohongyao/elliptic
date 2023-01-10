import torch
import torch.nn.functional as F
import torch.nn as nn
import models  # our model
from log import get_logger
import argparse
import json
import os
import random
import shutil
from utils import metrics, get_datasets
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from log import get_logger

parser = argparse.ArgumentParser(description='Elliptic Training')
parser.add_argument('--gpu', default="", type=str, help='GPU id to use.')
parser.add_argument('-a', '--arch', default='gcn', help='model architecture')
parser.add_argument('-s', '--suffix', default='', help='suffix')
parser.add_argument('-e',
                    '--epochs',
                    default=1000,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.005,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--nc',
                    '--num-classes',
                    default=2,
                    type=int,
                    help='num class')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=5e-4,
                    type=float,
                    help='weight decay (mini_net_sgd: 5e-4)',
                    dest='weight_decay')
parser.add_argument('--seed',
                    default=0,
                    type=int,
                    help='seed for initializing training. ')
args = parser.parse_args()
if args.gpu != "":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(state, is_best, base_dir):
    filename = os.path.join(base_dir, 'last.pth')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(base_dir, 'best.pth'))


def train(model, data, train_idx, optimizer):
    # data.y is labels of shape (N, )
    anomaly_train = torch.tensor(list(
        set(torch.argwhere(data.y == 1).squeeze(1).tolist())
        & set(train_idx.tolist())),
                                 dtype=torch.long).cuda()
    normal_train = torch.tensor(list(
        set(torch.argwhere(data.y == 0).squeeze(1).tolist())
        & set(train_idx.tolist())),
                                dtype=torch.long).cuda()
    extra_args = dict(anomaly_train=anomaly_train, normal_train=normal_train)
    model.train()
    out = model(data.x, data.edge_index, **extra_args)
    extra_loss = None
    if isinstance(out, tuple):
        out, extra_loss = out
        out = out[train_idx]
    else:
        out = out[train_idx]
    loss = F.cross_entropy(out, data.y[train_idx])
    if extra_loss is not None:
        loss = loss + extra_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    results = metrics(data.y[train_idx], out)
    return results, loss.item()


def test(model, data, test_idx):
    #
    with torch.no_grad():
        model.eval()
        out = model(data.x, data.edge_index)[test_idx]
        losses = F.cross_entropy(out, data.y[test_idx]).item()
        results = metrics(data.y[test_idx], out)
    return results, losses


def main():
    if args.suffix != '':
        args.suffix = '_' + args.suffix
    best_metrics = {}
    log_dir = f'run/{args.arch}{args.suffix}'
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "env.json"), "w") as f:
        json.dump(vars(args), f)
    logger = get_logger(log_dir, f'train.log', resume="", is_rank0=True)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=60)
    data = get_datasets().cuda()
    model = getattr(models, args.arch)(in_feats=data.x.size(-1),
                                       out_feats=2).cuda()

    logger.info(
        f"Model {args.arch} Numel {sum(p.numel() for p in model.parameters())}"
    )  #模型总参数量
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, verbose=True)
    train_idx, val_idx = data.train_idx, data.val_idx
    for epoch in range(args.epochs):
        logger.info(f'Train/Epoch {epoch}/{args.epochs}')
        results, loss = train(model, data, train_idx, optimizer)

        logger.info('Train/loss %.5f' % (loss))
        writer.add_scalar('Train/loss', loss, epoch)
        logger.info(f'Train/metrics {results}')
        for k, v in results.items():
            writer.add_scalar(f'Train/{k}', v, epoch)

        logger.info(f'Val/Epoch {epoch}/{args.epochs}')
        results, loss = test(model, data, val_idx)

        logger.info('Val/loss %.5f' % (loss))
        writer.add_scalar('Val/loss', loss, epoch)
        logger.info(f'Val/metrics {results}')
        for k, v in results.items():
            writer.add_scalar(f'Val/{k}', v, epoch)
        scheduler.step()

        is_best = {}
        for k, v in results.items():
            if k not in best_metrics or best_metrics[k] < v:
                best_metrics[k] = v
                is_best[k] = True
            else:
                is_best[k] = False
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_metrics': best_metrics,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best['f1'], log_dir)
        logger.info(f'Checkpoint {epoch} saved ! {is_best}')


if __name__ == '__main__':
    setup_seed(args.seed)
    main()