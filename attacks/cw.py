import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
import sys
sys.path.append('..')

from utils import AverageMeter, accuracy_top1, accuracy
from attacks.step import LinfStep, L2Step

STEPS = {
    'Linf': LinfStep,
    'L2': L2Step,
}


def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

class CWLoss(nn.Module):
    def __init__(self, num_classes=10, margin=50, reduced=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduced = reduced
        return

    def forward(self, logits, targets):
        onehot_targets = one_hot_tensor(targets, self.num_classes,
                                        targets.device)

        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduced:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss

def batch_adv_attack(args, model, x, target):
    orig_x = x.clone().detach()
    step = STEPS[args.constraint](orig_x, args.eps, args.step_size)

    @torch.enable_grad()
    def get_adv_examples(x):
        for _ in range(args.num_steps):
            x = x.clone().detach().requires_grad_(True)
            logits = model(x)
            loss = -1 * CWLoss()(logits, target)
            grad = torch.autograd.grad(loss, [x])[0]
            with torch.no_grad():
                x = step.step(x, grad)
                x = step.project(x)
                x = torch.clamp(x, 0, 1)
        return x.clone().detach()
    
    to_ret = None

    if args.random_restarts == 0:
        adv = get_adv_examples(x)
        to_ret = adv.detach()
    elif args.random_restarts == 1:
        x = step.random_perturb(x)
        x = torch.clamp(x, 0, 1)
        adv = get_adv_examples(x)
        to_ret = adv.detach()
    else:
        for _ in range(args.random_restarts):
            x = step.random_perturb(x)
            x = torch.clamp(x, 0, 1)

            adv = get_adv_examples(x)
            if to_ret is None:
                to_ret = adv.detach()
            
            logits = model(adv)
            corr, = accuracy(logits, target, topk=(1,), exact=True)
            corr = corr.bool()
            misclass = ~corr
            to_ret[misclass] = adv[misclass]
    
    return to_ret.detach().requires_grad_(False)


@torch.no_grad()
def cw_attack(args, model, loader, writer=None, epoch=0, loop_type='test'):
    model.eval()
    loss_logger = AverageMeter()
    acc_logger = AverageMeter()
    ATTACK_NAME = 'CW-{}'.format(args.num_steps)

    iterator = tqdm(enumerate(loader), total=len(loader), ncols=110)
    for i, (inp, target) in iterator:
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        inp_adv = batch_adv_attack(args, model, inp, target)
        logits = model(inp_adv)

        loss = nn.CrossEntropyLoss()(logits, target)
        acc = accuracy_top1(logits, target)

        loss_logger.update(loss.item(), inp.size(0))
        acc_logger.update(acc, inp.size(0))

        desc = ('[{} {}] | Loss {:.4f} | Accuracy {:.4f} ||'
                .format(ATTACK_NAME, loop_type, loss_logger.avg, acc_logger.avg))
        iterator.set_description(desc)

    if writer is not None:
        descs = ['loss', 'accuracy']
        vals = [loss_logger, acc_logger]
        for k, v in zip(descs, vals):
            writer.add_scalar('adv_{}_{}'.format(loop_type, k), v.avg, epoch)

    return loss_logger.avg, acc_logger.avg, ATTACK_NAME






