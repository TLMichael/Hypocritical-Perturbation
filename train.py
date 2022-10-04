import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import AverageMeter, accuracy_top1
from attacks.natural import natural_attack
from attacks.adv import adv_attack, batch_adv_attack
from attacks.trades import batch_trades_attack


def standard_loss(args, model, x, y):
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    return loss, logits

def adv_loss(args, model, x, y):
    model.eval()
    x_adv = batch_adv_attack(args, model, x, y)
    model.train()

    logits_adv = model(x_adv)
    loss = nn.CrossEntropyLoss()(logits_adv, y)
    return loss, logits_adv

def trades_loss(args, model, x, y, beta=6.0):
    model.eval()
    x_adv = batch_trades_attack(args, model, x, y)
    model.train()

    logits = model(torch.cat((x, x_adv), dim=0))
    logits_cln, logits_adv = logits[:logits.size(0)//2], logits[logits.size(0)//2:]
    kl = nn.KLDivLoss(reduction='batchmean')

    loss_rob = kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_cln, dim=1))
    loss_nat = nn.CrossEntropyLoss()(logits_cln, y)
    loss = loss_nat + beta * loss_rob
    return loss, logits_cln

def mart_loss(args, model, x_natural, y, beta=6.0):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if args.constraint == 'Linf':
        for _ in range(args.num_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + args.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - args.eps), x_natural + args.eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    x_adv = torch.clamp(x_adv, 0.0, 1.0).clone().detach()
    logits = model(x_natural)
    logits_adv = model(x_adv)
    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    nat_probs = F.softmax(logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + beta * loss_robust
    return loss, logits_adv

LOSS_FUNC = {
    '': standard_loss,
    'ST': standard_loss,
    'AT': adv_loss,
    'TRADES': trades_loss,
    'MART': mart_loss,
}

def train(args, model, optimizer, loader, writer, epoch):
    model.train()
    loss_logger = AverageMeter()
    acc_logger = AverageMeter()

    iterator = tqdm(enumerate(loader), total=len(loader), ncols=95)
    for i, (inp, target) in iterator:
        inp = inp.cuda()
        target = target.cuda()

        loss, logits = LOSS_FUNC[args.train_loss](args, model, inp, target)
        acc = accuracy_top1(logits, target)

        loss_logger.update(loss.item(), inp.size(0))
        acc_logger.update(acc, inp.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        desc = 'Train Epoch: {} | Loss {:.4f} | Accuracy {:.4f} ||'.format(epoch, loss_logger.avg, acc_logger.avg)
        iterator.set_description(desc)

    if writer is not None:
        descs = ['loss', 'accuracy']
        vals = [loss_logger, acc_logger]
        for d, v in zip(descs, vals):
            writer.add_scalar('train_{}'.format(d), v.avg, epoch)

    return loss_logger.avg, acc_logger.avg

def train_model(args, model, optimizer, schedule, train_loader, val_loader, test_loader, writer):
    if args.epochs == 0:
        checkpoint = {
                'model': model.state_dict(),
                'epoch': 0,
                'train_acc': -1,
                'train_loss': -1,
                'cln_val_acc': -1,
                'cln_val_loss': -1,
                'cln_test_acc': -1,
                'cln_test_loss': -1,
                'adv_val_acc': -1,
                'adv_val_loss': -1,
                'adv_test_acc': -1,
                'adv_test_loss': -1,
            }
        torch.save(checkpoint, args.model_path)
        torch.save(checkpoint, args.model_path_last)

    best_acc = 0.
    for epoch in range(args.epochs):
        train_loss, train_acc = train(args, model, optimizer, train_loader, writer, epoch)

        last_epoch = (epoch == (args.epochs - 1))
        should_log = (epoch % args.log_gap == 0)

        if should_log or last_epoch:
            cln_val_loss, cln_val_acc, _ = natural_attack(args, model, val_loader, writer, epoch, 'val')
            cln_test_loss, cln_test_acc, _ = natural_attack(args, model, test_loader, writer, epoch, 'test')

            robust_target = (args.train_loss in ['AT', 'TRADES', 'MART'])
            if robust_target:
                adv_val_loss, adv_val_acc, _ = adv_attack(args, model, val_loader, writer, epoch, 'val')
                adv_test_loss, adv_test_acc, _ = adv_attack(args, model, test_loader, writer, epoch, 'test')
                our_acc = adv_val_acc
            else:
                adv_val_loss, adv_val_acc, adv_test_loss, adv_test_acc = -1, -1, -1, -1
                our_acc = cln_val_acc

            is_best = our_acc > best_acc
            best_acc = max(our_acc, best_acc)

            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'cln_val_acc': cln_val_acc,
                'cln_val_loss': cln_val_loss,
                'cln_test_acc': cln_test_acc,
                'cln_test_loss': cln_test_loss,
                'adv_val_acc': adv_val_acc,
                'adv_val_loss': adv_val_loss,
                'adv_test_acc': adv_test_acc,
                'adv_test_loss': adv_test_loss,
                
            }
            if is_best:
                torch.save(checkpoint, args.model_path)
            torch.save(checkpoint, args.model_path_last)
        schedule.step()
    return model

def eval_model(args, model, test_loader):
    model.eval()
    args.eps = 8/255

    keys, values = [], []
    keys.append('Model')
    values.append(args.model_path)

    # Natural
    _, acc, name = natural_attack(args, model, test_loader)
    keys.append(name)
    values.append(acc)

    # FGSM
    args.num_steps = 1
    args.step_size = args.eps
    args.random_restarts = 0
    _, acc, name = adv_attack(args, model, test_loader)
    keys.append('FGSM')
    values.append(acc)

    # PGD-20
    args.num_steps = 20
    args.step_size = args.eps / 4
    args.random_restarts = 1
    _, acc, name = adv_attack(args, model, test_loader)
    keys.append(name)
    values.append(acc)

    # PGD-100
    args.num_steps = 100
    args.step_size = args.eps / 4
    args.random_restarts = 1
    _, acc, name = adv_attack(args, model, test_loader)
    keys.append(name)
    values.append(acc)

    # CW-100
    from attacks.cw import cw_attack
    args.num_steps = 100
    args.step_size = args.eps / 4
    args.random_restarts = 1
    _, acc, name = cw_attack(args, model, test_loader)
    keys.append(name)
    values.append(acc)

    # AutoAttack
    from autoattack import AutoAttack
    adversary = AutoAttack(model, norm=args.constraint, eps=args.eps, version='standard')
    x_test = torch.cat([x for (x, y) in test_loader])
    y_test = torch.cat([y for (x, y) in test_loader])
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)
    auto_acc = adversary.clean_accuracy(x_adv, y_test, bs=args.batch_size) * 100
    keys.append('AotuAttack')
    values.append(auto_acc)
    
    # Save results
    import csv
    csv_fn = '{}.csv'.format(args.model_path)
    with open(csv_fn, 'w') as f:
        write = csv.writer(f)
        write.writerow(keys)
        write.writerow(values)

    print('=> csv file is saved at [{}]'.format(csv_fn))