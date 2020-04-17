import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.autograd import Variable
from torchvision import transforms

import deeplab
from pascal import VOCSegmentation
from cityscapes import Cityscapes
from multiorgan import Multiorgan
from utils import AverageMeter, inter_and_union

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False,
                    help='training mode')
parser.add_argument('--exp', type=str, required=True,
                    help='name of experiment')
parser.add_argument('--gpu', type=int, default=0,
                    help='test time gpu device id')
parser.add_argument('--backbone', type=str, default='resnet101',
                    help='resnet101')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='pascal or cityscapes or multiorgan')
parser.add_argument('--groups', type=int, default=None, 
                    help='num of groups for group normalization')
parser.add_argument('--epochs', type=int, default=100,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch size')
parser.add_argument('--base_lr', type=float, default=0.00002,
                    help='base learning rate')
parser.add_argument('--last_mult', type=float, default=1.0,
                    help='learning rate multiplier for last layers')
parser.add_argument('--scratch', action='store_true', default=False,
                    help='train from scratch')
parser.add_argument('--freeze_bn', action='store_true', default=False,
                    help='freeze batch normalization parameters')
parser.add_argument('--weight_std', action='store_true', default=False,
                    help='weight standardization')
parser.add_argument('--beta', action='store_true', default=False,
                    help='resnet101 beta')
parser.add_argument('--crop_size', type=int, default=512,
                    help='image crop size')
parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint to resume from')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers')
args = parser.parse_args()


def main():
  assert torch.cuda.is_available()
  torch.backends.cudnn.benchmark = True
  exp_dir = os.path.join('result/{}'.format(args.exp))
  if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir)
  models_dir = os.path.join(exp_dir, 'models')
  if not os.path.isdir(models_dir):
    os.makedirs(models_dir)

  model_fname = os.path.join(models_dir, 'deeplab_{0}_{1}_v3_{2}_epoch%d.pth'.format(
      args.backbone, args.dataset, args.exp))

  if args.dataset == 'pascal':
    dataset = VOCSegmentation('data/VOCdevkit',
        train=args.train, crop_size=args.crop_size)
  elif args.dataset == 'cityscapes':
    dataset = Cityscapes('data/cityscapes',
        train=args.train, crop_size=args.crop_size)
  elif args.dataset == 'multiorgan':
    dataset = Multiorgan('data/multiorgan',
        train=args.train, crop_size=args.crop_size,dataset=args.dataset)
  else:
    raise ValueError('Unknown dataset: {}'.format(args.dataset))
  if args.backbone == 'resnet101':
    model = getattr(deeplab, 'resnet101')(
        pretrained=(not args.scratch),
        num_classes=len(dataset.CLASSES),
        num_groups=args.groups,
        weight_std=args.weight_std,
        beta=args.beta)
  else:
    raise ValueError('Unknown backbone: {}'.format(args.backbone))

  if args.train:
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    model = nn.DataParallel(model).cuda()
    model.train()
    if args.freeze_bn:
      for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
          m.eval()
          m.weight.requires_grad = False
          m.bias.requires_grad = False
    backbone_params = (
        list(model.module.conv1.parameters()) +
        list(model.module.bn1.parameters()) +
        list(model.module.layer1.parameters()) +
        list(model.module.layer2.parameters()) +
        list(model.module.layer3.parameters()) +
        list(model.module.layer4.parameters()))
    last_params = list(model.module.aspp.parameters())
    optimizer = optim.SGD([
      {'params': filter(lambda p: p.requires_grad, backbone_params)},
      {'params': filter(lambda p: p.requires_grad, last_params)}],
      lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.train,
        pin_memory=True, num_workers=args.workers)
    max_iter = args.epochs * len(dataset_loader)
    losses = AverageMeter()
    start_epoch = 0

    if args.resume:
      if os.path.isfile(args.resume):
        print('=> loading checkpoint {0}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> loaded checkpoint {0} (epoch {1})'.format(
          args.resume, checkpoint['epoch']))
      else:
        print('=> no checkpoint found at {0}'.format(args.resume))

    loss_list_train = []
    for epoch in range(start_epoch, args.epochs):
      for i, (inputs, target) in enumerate(dataset_loader):
        cur_iter = epoch * len(dataset_loader) + i
        lr = args.base_lr * (1 - float(cur_iter) / max_iter) ** 0.9
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr * args.last_mult

        inputs = Variable(inputs.cuda())
        # print(inputs.data.shape)
        target = Variable(target.cuda())
        # print(target.data.shape)
        # print(inputs.data.shape)
        # target_np = target.data.cpu().numpy()
        # print(np.unique(target_np))
        outputs = model(inputs)

        # print(outputs.data.shape)
        loss = criterion(outputs, target)
        # print(loss)
        if np.isnan(loss.item()) or np.isinf(loss.item()):
          pdb.set_trace()
        losses.update(loss.item(), args.batch_size)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print('epoch: {0}\t'
              'iter: {1}/{2}\t'
              'lr: {3:.6f}\t'
              'loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
              epoch + 1, i + 1, len(dataset_loader), lr, loss=losses))
      loss_list_train.append(losses.ema)

      if epoch % 1 == 0:
        torch.save({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          }, model_fname % (epoch + 1))

    # save valdiation/training loss to pick best model
    train_loss_file = os.path.join(exp_dir, 'train_loss.txt')
    train_wr = open(train_loss_file, 'w')
    for loss_train in loss_list_train:
      train_wr.write(str(loss_train) + '\n')
    train_wr.close()


  else:
        # validation, store loss and dice for stopping creteria
    colorpick = [[0,0,0], [255,30,30],[255,245,71],[112,255,99],[9,150,37],[30,178,252],[132,0,188],\
        [255,81,255],[158,191,9],[255,154,2],[102,255,165],[0,242,209],[255,0,80],[255,0,160],[100,100,100],[170,170,170],[230,230,230]]
    #Spleen: red,right kid: yellow, left kid green, gall:sky blue, eso:blue,liver:lg blue
    #sto:pink,aorta: purple,IVC, potal vein: orange, pancreas: favor, adrenal gland
    organ_list = ['bk','spleen', 'right_kigney', 'left_kidney', 'gallbladder', 'esophagus', 'liver',\
                    'stomach', 'aorta', 'IVC', 'veins', 'pancreas', 'r_adrenal_gland','l_adrenal_gland', 'body','bone','lung']
    loss_list_val = []

    for epoch in range(70, args.epochs+1, 10):
      losses = AverageMeter()
      with torch.no_grad():
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
        model.eval()
        checkpoint = torch.load(model_fname % (epoch))
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
        model.load_state_dict(state_dict)

        # cmap = loadmat('data/pascal_seg_colormap.mat')['colormap']
        # print(type(cmap))
        colorpick = np.array(colorpick)
        cmap = colorpick.flatten().tolist()
        # print(cmap)
        for i, (inputs, target) in enumerate(dataset):
          inputs, target = dataset[i]
          inputs = Variable(inputs.cuda())
          # print(inputs.unsqueeze(0).data.shape)
          outputs = model(inputs.unsqueeze(0))


          _, pred = torch.max(outputs, 1)
          pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
          mask = target.data.cpu().numpy().astype(np.uint8)
          imname = dataset.masks[i].split('/')[-1]
          mask_pred = Image.fromarray(pred)
          mask_pred.putpalette(cmap)
          outlabel_dir = os.path.join(exp_dir, 'val_{}'.format(epoch))
          if not os.path.isdir(outlabel_dir):
            os.makedirs(outlabel_dir)
          mask_pred.save(os.path.join(outlabel_dir, imname))
          print('testing: {}'.format(imname))

    # with torch.no_grad():
    #   torch.cuda.set_device(args.gpu)
    #   model = model.cuda()
    #   model.eval()
    #   checkpoint = torch.load(model_fname % args.epochs)
    #   state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    #   model.load_state_dict(state_dict)
    #   cmap = loadmat('data/pascal_seg_colormap.mat')['colormap']
    #   cmap = (cmap * 255).astype(np.uint8).flatten().tolist()

    #   inter_meter = AverageMeter()
    #   union_meter = AverageMeter()
    #   for i in range(len(dataset)):
    #     inputs, target = dataset[i]
    #     inputs = Variable(inputs.cuda())
    #     print(inputs.data.shape)
    #     outputs = model(inputs.unsqueeze(0))
    #     print(outputs.data.shape)

    #     _, pred = torch.max(outputs, 1)
    #     pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
    #     mask = target.numpy().astype(np.uint8)
    #     imname = dataset.masks[i].split('/')[-1]
    #     mask_pred = Image.fromarray(pred)
    #     mask_pred.putpalette(cmap)
    #     mask_pred.save(os.path.join('data/val', imname))
    #     print('eval: {0}/{1}'.format(i + 1, len(dataset)))

    #     inter, union = inter_and_union(pred, mask, len(dataset.CLASSES))
    #     inter_meter.update(inter)
    #     union_meter.update(union)

    #   iou = inter_meter.sum / (union_meter.sum + 1e-10)
    #   for i, val in enumerate(iou):
    #     print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
    #   print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))


if __name__ == "__main__":
  main()
