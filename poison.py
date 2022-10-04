import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import argparse
import os
import torchvision
from tqdm import tqdm
from pprint import pprint

from utils import set_seed, PoisonDataset, make_and_restore_cifar_model, AverageMeter, accuracy_top1, show_image_row, cifar10_class
from utils import infer_poison_name, infer_exp_name
from attacks.step import LinfStep, L2Step

STEPS = {
    'Linf': LinfStep,
    'L2': L2Step,
}


from utils import RandomTransform
params = dict(source_size=32, target_size=32, shift=8, fliplr=True)
trans = RandomTransform(**params, mode='bilinear')


def batch_poison(model, x, target, args):
    orig_x = x.clone().detach()
    step = STEPS[args.constraint](orig_x, args.eps, args.step_size)

    if args.poison_type == 'Adv':
        target = (target + 1) % args.num_classes  # Error-maximizing noise: Using a fixed permutation of labels
    elif args.poison_type == 'Hyp':
        target = target     # Error-minimizing noise

    for _ in range(args.poison_steps):
        x = x.clone().detach().requires_grad_(True)
        if args.poison_aug == True:
            x_aug = trans(x)
            logits = model(x_aug)
        else:
            logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, target)
        grad = torch.autograd.grad(loss, [x])[0]
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
            x = torch.clamp(x, 0, 1)
    
    return x.clone().detach().requires_grad_(False)

def crafting_poison(args, loader, model):
    poisoned_input = []
    clean_target = []
    loss_logger = AverageMeter()
    acc_logger = AverageMeter()
    iterator = tqdm(enumerate(loader), total=len(loader))
    for i, (inp, target) in iterator:
        inp, target = inp.cuda(), target.cuda()
        inp_p = batch_poison(model, inp, target, args)
        poisoned_input.append(inp_p.detach().cpu())
        clean_target.append(target.detach().cpu())
        with torch.no_grad():
            logits = model(inp_p)
            loss = nn.CrossEntropyLoss()(logits, target)
            acc = accuracy_top1(logits, target)
        loss_logger.update(loss.item(), inp.size(0))
        acc_logger.update(acc, inp.size(0))
        desc = ('[{} {:.3f}] | Loss {:.4f} | Accuracy {:.3f} ||'
                .format(args.poison_name, args.eps, loss_logger.avg, acc_logger.avg))
        iterator.set_description(desc)
    poisoned_input = torch.cat(poisoned_input, dim=0)
    clean_target = torch.cat(clean_target, dim=0)
    return poisoned_input, clean_target

def visualize(args, clean_loader, poison_loader):
    clean_iterator = iter(clean_loader)
    poison_iterator = iter(poison_loader)
    for i in range(1):
        clean_inp, clean_label = next(clean_iterator)
        poison_inp, poison_label = next(poison_iterator)

        show_image_row([clean_inp], tlist=[[args.classes[int(t)] for t in clean_label]], fontsize=20, filename=args.poison_path+'.{}ori.png'.format(i))
        show_image_row([poison_inp], tlist=[[args.classes[int(t)] for t in poison_label]], fontsize=20, filename=args.poison_path+'.{}.png'.format(i))


def main(args):
    if args.poison_type == 'Clean':
        print('Natural adversarial examples already exist.')
        return
    if os.path.isfile(args.poison_path):
        print('Poison [{}] already exists.'.format(args.poison_path))
        return
    
    data_set = datasets.CIFAR10(args.data_path, train=True, transform=transforms.ToTensor())
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)
    model = make_and_restore_cifar_model(args.craft_model_arch, resume_path=args.model_path)
    model.eval()

    set_seed(args.seed)
    poison_data = crafting_poison(args, data_loader, model)
    torch.save(poison_data, args.poison_path)

    poison_set = PoisonDataset(args.poison_path, transforms.ToTensor())
    poison_loader = DataLoader(poison_set, batch_size=5, shuffle=False)
    clean_loader = DataLoader(data_set, batch_size=5, shuffle=False)
    visualize(args, clean_loader, poison_loader)

def main_vis(args):
    data_set = datasets.CIFAR10(args.data_path, train=True, transform=transforms.ToTensor())
    poison_set = PoisonDataset(args.poison_path, transforms.ToTensor())
    poison_loader = DataLoader(poison_set, batch_size=5, shuffle=False)
    clean_loader = DataLoader(data_set, batch_size=5, shuffle=False)
    
    clean_iterator = iter(clean_loader)
    poison_iterator = iter(poison_loader)
    for i in range(1):
        clean_inp, clean_label = next(clean_iterator)
        poison_inp, poison_label = next(poison_iterator)
        for j in range(len(clean_inp)):
            torchvision.utils.save_image(clean_inp[j], padding=1, fp=args.poison_path+'.clean_{}.png'.format(j))
            torchvision.utils.save_image(poison_inp[j], padding=1, fp=args.poison_path+'.perturb_{}.png'.format(j))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generating poisons for CIFAR10')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--out_dir', default='results/CIFAR10', type=str)

    parser.add_argument('--eps', default=8, type=float)
    parser.add_argument('--constraint', default='Linf', type=str, choices=['Linf', 'L2'])

    parser.add_argument('--poison_type', default='Hyp', type=str, choices=['Clean', 'Adv', 'Hyp'])
    parser.add_argument('--poison_steps', default=100, type=int)
    parser.add_argument('--poison_aug', default=True, type=bool)
    parser.add_argument('--craft_model_loss', default='AT', type=str, choices=['ST', 'AT'])
    parser.add_argument('--craft_model_eps', default=2, type=float)
    parser.add_argument('--craft_model_epoch', default=10, type=int)
    parser.add_argument('--craft_model_arch', default='ResNet18', type=str)

    args = parser.parse_args()

    # Crafting options
    args.eps = args.eps / 255
    args.craft_model_eps = args.craft_model_eps / 255
    args.step_size = args.eps / 10
    args.batch_size = 256
    args.classes = cifar10_class
    args.num_classes = 10

    # Miscellaneous
    args.data_path = '../datasets/CIFAR10'
    args.exp_name = infer_exp_name(args.craft_model_loss, args.craft_model_eps, args.craft_model_epoch, args.craft_model_arch, 'Clean')
    args.model_path = os.path.join(args.out_dir, args.exp_name, 'checkpoint_last.pth')
    args.poison_name = infer_poison_name(args.poison_type, args.poison_steps, args.craft_model_loss, args.craft_model_eps, args.craft_model_epoch, args.craft_model_arch, args.poison_aug)
    args.poison_path = os.path.join(args.out_dir, args.exp_name, args.poison_name + '.poison')
    
    pprint(vars(args))

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    main(args)