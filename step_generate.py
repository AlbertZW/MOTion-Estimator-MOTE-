import os
import time
import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops.logger import setup_logger
from ops.lr_scheduler import get_scheduler
from ops.utils import reduce_tensor
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from tensorboardX import SummaryWriter
from torch.utils.data import *
import torchvision

from ops.ssim import msssim

best_prec1 = 0

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

def main():
    global args, best_prec1
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    model_name = 'MOTE_nodense'

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset, args.modality)
    full_arch_name = args.arch
    args.store_name = '_'.join([model_name, args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments, 'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)

    if dist.get_rank() == 0:
        check_rootfolders()

    logger = setup_logger(output=os.path.join(args.root_log, args.store_name),
                          distributed_rank=dist.get_rank(),
                          name=f'TDN')
    logger.info('storing name: ' + args.store_name)

    teacher_model = TSN(num_class,
                        args.num_segments,
                        args.modality,
                        base_model='resnet50',
                        consensus_type=args.consensus_type,
                        dropout=args.dropout,
                        img_feature_dim=args.img_feature_dim,
                        partial_bn=not args.no_partialbn,
                        pretrain=args.pretrain,
                        fc_lr5=(args.tune_from and args.dataset in args.tune_from),
                        step=-1)
    print('=> loading checkpoint for teacher network.')
    teacher_ckpt_path = os.path.join(args.root_model, 'Teacher__' + args.dataset + '/ckpt_best.pth')
    t_checkpoint = torch.load(teacher_ckpt_path)
    sd = t_checkpoint['state_dict']
    model_dict = teacher_model.state_dict()
    replace_dict = {}
    for k, v in sd.items():
        if k.startswith('module.base_model.'):
            tmp_k = k.replace('module.base_model.', 'base_model.')
            if tmp_k in model_dict:
                replace_dict[tmp_k] = v
                # print(tmp_k)
    model_dict.update(replace_dict)
    teacher_model.load_state_dict(model_dict)
    teacher_model = DistributedDataParallel(teacher_model.cuda(), device_ids=[args.local_rank], broadcast_buffers=True,
                            find_unused_parameters=True)

    model = TSN(num_class,
                args.num_segments,
                args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                fc_lr5=(args.tune_from and args.dataset in args.tune_from),
                step=1)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    for group in policies:
        logger.info(
            ('[' + model_name + '-{}]group: {} has {} params, lr_mult: {}, decay_mult: {}'.
             format(args.arch, group['name'], len(group['params']),
                    group['lr_mult'], group['decay_mult'])))

    train_augmentation = model.get_augmentation(
        flip=False if 'something' in args.dataset else True)

    cudnn.benchmark = True

    # Data loading code
    normalize = GroupNormalize(input_mean, input_std)

    train_dataset = TSNDataSet(
        args.dataset,
        args.root_path,
        args.train_list,
        num_segments=args.num_segments,
        new_length=3,
        modality=args.modality,
        image_tmpl=prefix,
        transform=torchvision.transforms.Compose([train_augmentation,
            Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
            normalize,]),
        dense_sample=args.dense_sample)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, sampler=train_sampler,drop_last=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        mse_criterion = torch.nn.MSELoss(reduce=True, size_average=True).cuda()
        ssim_criterion = msssim
    else:
        raise ValueError("Unknown loss type")

    # optimizer = torch.optim.SGD(policies, args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(policies, args.lr, weight_decay=args.weight_decay)

    scheduler = get_scheduler(optimizer, len(train_loader), args)

    model = DistributedDataParallel(model.cuda(), device_ids=[args.local_rank], broadcast_buffers=True, find_unused_parameters=True)

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info(("=> loaded checkpoint '{}' (epoch {})".format(
                args.evaluate, checkpoint['epoch'])))
        else:
            logger.info(("=> no checkpoint found at '{}'".format(args.resume)))


    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))

    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    if args.evaluate:
        print("No Evaluation supported during generating step.")
        return

    for epoch in range(args.start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_loss = train(train_loader, teacher_model, model, mse_criterion, ssim_criterion, optimizer, epoch=epoch,
                           logger=logger, scheduler=scheduler)

        if dist.get_rank() == 0:
            tf_writer.add_scalar('loss/train', train_loss, epoch)
            tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = prec5 = val_loss = 0

            if dist.get_rank() == 0:
                tf_writer.add_scalar('loss/test', val_loss, epoch)
                tf_writer.add_scalar('acc/test_top1', prec1, epoch)
                tf_writer.add_scalar('acc/test_top5', prec5, epoch)

                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

                logger.info(("Best Prec@1: '{}'".format(best_prec1)))
                tf_writer.flush()
                save_epoch = epoch + 1
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'prec1': prec1,
                        'best_prec1': best_prec1,
                    }, save_epoch, is_best)


def train(train_loader, teacher_model, model, MSE_criterion, ssim_criterion, optimizer, epoch,
          logger=None, scheduler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    mse_losses = AverageMeter()
    ssim_losses = AverageMeter()
    losses = AverageMeter()


    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    model.train()

    end = time.time()

    loss_recorder_path = os.path.join(args.root_model, args.store_name, 'step_generate_losses.txt')
    with open(loss_recorder_path, 'a') as recorder:
        recorder.write('Epoch: ' + str(epoch) + '\n')

    for i, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end)
        target = target.cuda()
        input_var = input.cuda()
        with torch.no_grad():
            tdn_motion = teacher_model(input_var)

        generated_motion = model(input_var)
        mse_loss = MSE_criterion(generated_motion, tdn_motion)

        ssim_loss = 1 - ssim_criterion(generated_motion, tdn_motion, normalize=True)

        loss = mse_loss + 0 * ssim_loss

        mse_losses.update(mse_loss.item(), input.size(0))
        ssim_losses.update(ssim_loss.item(), input.size(0))
        losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()

        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        scheduler.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'Loss {mse.val:.4f}+{ssim.val:.4f}={loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                mse=mse_losses, ssim=ssim_losses, loss=losses, lr=optimizer.param_groups[-1]['lr'])))  # TODO

    return losses.avg


def save_checkpoint(state, epoch, is_best):
    filename = '%s/%s/ckpt_step1_epoch%s.pth' % (args.root_model, args.store_name, epoch)
    torch.save(state, filename)
    if is_best:
        torch.save(state, filename.replace('ckpt', 'ckpt_best'))


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [
        args.root_log, args.root_model,
        os.path.join(args.root_log, args.store_name),
        os.path.join(args.root_model, args.store_name)
    ]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder)


if __name__ == '__main__':
    main()