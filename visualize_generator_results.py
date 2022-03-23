# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

import os
import time
import torch
import torch.nn.parallel
# from torch.nn.parallel import DistributedDataParallel
# import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from ops.dataset import TSNDataSet
from visualization.models import TSN
from ops.transforms import *
from ops.logger import setup_logger
from ops.lr_scheduler import get_scheduler
from ops.utils import reduce_tensor
# from ops.Gan_modules import d_logistic_loss, d_r1_loss

from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from tensorboardX import SummaryWriter
from torch.utils.data import *
import torchvision

# import cv2
import matplotlib.pyplot as plt

torch.cuda.manual_seed(26)


def select_visual_samples(train_list, val_list, vis_sample_list, sample_num=(15, 5)):
    if os.path.exists(vis_sample_list):
        return

    list_writer = open(vis_sample_list, 'a')
    with open(train_list) as tlf:
        lines = tlf.readlines()
        samples = random.sample(lines, sample_num[0])
        for sample in samples:
            sample = sample.replace('\n', '')
            list_writer.write(sample + '\n')
    with open(val_list) as vlf:
        lines = vlf.readlines()
        samples = random.sample(lines, sample_num[1])
        for sample in samples:
            sample = sample.replace('\n', '')
            list_writer.write(sample + '\n')
    list_writer.close()


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


def main():
    args = parser.parse_args()

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset, args.modality)
    full_arch_name = args.arch + '_' + args.consensus_type

    store_name = '_'.join([args.dataset, args.modality, full_arch_name, 'segment%d' % args.num_segments, 'e{}'.format(args.epochs)])

    args.store_name = '_'.join(['MOTE_wSSIM', store_name])

    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)

    visualization_folder = os.path.join('./visualization', args.store_name)
    if not os.path.exists(visualization_folder):
        print('creating folder ' + visualization_folder)
        os.makedirs(visualization_folder)
    vis_sample_list = './visualization/vis_samples.txt'
    select_visual_samples(args.train_list, args.val_list, vis_sample_list)


    logger = setup_logger(output=os.path.join('./visualization', args.store_name), name=f'MGen_vis')
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
            step=0)

    # TDN_model = TSN(num_class,
    #         args.num_segments,
    #         args.modality,
    #         base_model='resnet50',
    #         consensus_type=args.consensus_type,
    #         dropout=args.dropout,
    #         img_feature_dim=args.img_feature_dim,
    #         partial_bn=not args.no_partialbn,
    #         pretrain=args.pretrain,
    #         fc_lr5=(args.tune_from and args.dataset in args.tune_from),
    #         step=-1)

    student_model = TSN(num_class,
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

    comp_model1 = TSN(num_class,
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

    comp_model2 = TSN(num_class,
            args.num_segments,
            args.modality,
            base_model=args.arch,
            consensus_type=args.consensus_type,
            dropout=args.dropout,
            img_feature_dim=args.img_feature_dim,
            partial_bn=not args.no_partialbn,
            pretrain=args.pretrain,
            fc_lr5=(args.tune_from and args.dataset in args.tune_from),
            step=-1)

    comp_model3 = TSN(num_class,
                      args.num_segments,
                      args.modality,
                      base_model=args.arch,
                      consensus_type=args.consensus_type,
                      dropout=args.dropout,
                      img_feature_dim=args.img_feature_dim,
                      partial_bn=not args.no_partialbn,
                      pretrain=args.pretrain,
                      fc_lr5=(args.tune_from and args.dataset in args.tune_from),
                      step=-2)

    crop_size = student_model.crop_size
    scale_size = student_model.scale_size
    input_mean = student_model.input_mean
    input_std = student_model.input_std
    policies = student_model.get_optim_policies()

    for group in policies:
        logger.info(
            ('[Motion_Gen-{}]group: {} has {} params, lr_mult: {}, decay_mult: {}'.
             format(args.arch, group['name'], len(group['params']),
                    group['lr_mult'], group['decay_mult'])))

    train_augmentation = student_model.get_augmentation(
        flip=False if 'something' in args.dataset else True)

    cudnn.benchmark = True

    # Data loading code
    normalize = GroupNormalize(input_mean, input_std)

    sample_length = 3

    dataset = TSNDataSet(
        args.dataset,
        args.root_path,
        vis_sample_list,
        num_segments=args.num_segments,
        new_length=sample_length,
        modality=args.modality,
        image_tmpl=prefix,
        random_shift=False,
        transform=torchvision.transforms.Compose([
            GroupScale(int(scale_size)), GroupCenterCrop(crop_size),
            Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
            normalize, ]),
        dense_sample=args.dense_sample)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    loader = torch.utils.data.DataLoader(dataset,
        batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, drop_last=True)


    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        # CE_criterion = torch.nn.CrossEntropyLoss().cuda()
        MSE_criterion = torch.nn.MSELoss(reduce=True, size_average=True).cuda()

    else:
        raise ValueError("Unknown loss type")


    print('=> loading checkpoint for single scale TDN network.')
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
    model_dict.update(replace_dict)
    teacher_model.load_state_dict(model_dict)
    teacher_model = torch.nn.DataParallel(teacher_model, device_ids=args.gpus).cuda()

    # print('=> loading checkpoint for multi scale TDN network.')
    # ms_ckpt_path = os.path.join(args.root_model, 'Teacher__' + args.dataset + '/best_.pth.tar')
    # ms_checkpoint = torch.load(ms_ckpt_path)
    # sd = ms_checkpoint['state_dict']
    # model_dict = TDN_model.state_dict()
    # replace_dict = {}
    # for k, v in sd.items():
    #     if k.startswith('module.base_model.'):
    #         tmp_k = k.replace('module.base_model.', 'base_model.')
    #         if tmp_k in model_dict:
    #             replace_dict[tmp_k] = v
    # model_dict.update(replace_dict)
    # TDN_model.load_state_dict(model_dict)
    # TDN_model = torch.nn.DataParallel(TDN_model, device_ids=args.gpus).cuda()


    visualization_folder = os.path.join(visualization_folder, 'final_vis')
    os.mkdir(visualization_folder)

    student_ckpt_path = os.path.join(args.root_model, args.store_name, 'ckpt_step1_epoch1.pth')
    if os.path.isfile(student_ckpt_path):
        logger.info(("=> loading checkpoint '{}'".format(student_ckpt_path)))
        checkpoint = torch.load(student_ckpt_path, map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        sd = checkpoint['state_dict']
        replace_dict = {}
        for k, v in sd.items():
            if k.startswith('module.base_model.'):
                tmp_k = k.replace('module.base_model.', 'base_model.')
                replace_dict[tmp_k] = v

        student_model.load_state_dict(replace_dict)
        logger.info(("=> loaded checkpoint '{}' (epoch {})".format(
            args.evaluate, checkpoint['epoch'])))
    else:
        logger.info(("=> no checkpoint found at '{}'".format(student_ckpt_path)))

    student_model = torch.nn.DataParallel(student_model, device_ids=args.gpus).cuda()

    comp1_ckpt_path = os.path.join(args.root_model, args.store_name, 'ckpt_step1_epoch0.pth')
    if os.path.isfile(comp1_ckpt_path):
        logger.info(("=> loading checkpoint '{}'".format(comp1_ckpt_path)))
        checkpoint = torch.load(comp1_ckpt_path, map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        sd = checkpoint['state_dict']
        replace_dict = {}
        for k, v in sd.items():
            if k.startswith('module.base_model.'):
                tmp_k = k.replace('module.base_model.', 'base_model.')
                replace_dict[tmp_k] = v

        comp_model1.load_state_dict(replace_dict)
        logger.info(("=> loaded checkpoint '{}' (epoch {})".format(
            args.evaluate, checkpoint['epoch'])))
    else:
        logger.info(("=> no checkpoint found at '{}'".format(comp1_ckpt_path)))

    comp_model1 = torch.nn.DataParallel(comp_model1, device_ids=args.gpus).cuda()

    comp2_ckpt_path = os.path.join(args.root_model, args.store_name, 'ckpt_step1_epoch-1.pth')
    if os.path.isfile(comp2_ckpt_path):
        logger.info(("=> loading checkpoint '{}'".format(comp2_ckpt_path)))
        checkpoint = torch.load(comp2_ckpt_path, map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        sd = checkpoint['state_dict']
        replace_dict = {}
        for k, v in sd.items():
            if k.startswith('module.base_model.'):
                tmp_k = k.replace('module.base_model.', 'base_model.')
                replace_dict[tmp_k] = v

        comp_model2.load_state_dict(replace_dict)
        logger.info(("=> loaded checkpoint '{}' (epoch {})".format(
            args.evaluate, checkpoint['epoch'])))
    else:
        logger.info(("=> no checkpoint found at '{}'".format(comp2_ckpt_path)))

    comp_model2 = torch.nn.DataParallel(comp_model2, device_ids=args.gpus).cuda()

    comp3_ckpt_path = os.path.join(args.root_model, args.store_name, 'ckpt_step1_epoch-2.pth')
    if os.path.isfile(comp3_ckpt_path):
        logger.info(("=> loading checkpoint '{}'".format(comp3_ckpt_path)))
        checkpoint = torch.load(comp3_ckpt_path, map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        sd = checkpoint['state_dict']
        replace_dict = {}
        for k, v in sd.items():
            if k.startswith('module.base_model.'):
                tmp_k = k.replace('module.base_model.', 'base_model.')
                replace_dict[tmp_k] = v

        comp_model3.load_state_dict(replace_dict)
        logger.info(("=> loaded checkpoint '{}' (epoch {})".format(
            args.evaluate, checkpoint['epoch'])))
    else:
        logger.info(("=> no checkpoint found at '{}'".format(comp2_ckpt_path)))

    comp_model3 = torch.nn.DataParallel(comp_model3, device_ids=args.gpus).cuda()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    teacher_model.eval()
    student_model.eval()

    if args.no_partialbn:
        student_model.module.partialBN(False)
    else:
        student_model.module.partialBN(True)

    end = time.time()

    for i, (input, target) in enumerate(loader):
        data_time.update(time.time() - end)
        input_var = input.cuda()

        with torch.no_grad():
            input = input.view((args.batch_size, args.num_segments, 9, 224, 224))
            input = input[:, :, 3:6, :, :]

            tdn_motion, fgd = teacher_model(input_var)
            generated_motion = student_model(input_var)

            comp_motion_1 = comp_model1(input_var)
            comp_motion_2 = comp_model2(input_var)
            comp_motion_3 = comp_model3(input_var)

            tdn_motion = tdn_motion.view((args.batch_size, args.num_segments) + tdn_motion.size()[-3:])
            generated_motion = generated_motion.view((args.batch_size, args.num_segments) + generated_motion.size()[-3:])

            comp_motion_1 = comp_motion_1.view((args.batch_size, args.num_segments) + comp_motion_1.size()[-3:])
            comp_motion_2 = comp_motion_2.view((args.batch_size, args.num_segments) + comp_motion_2.size()[-3:])
            comp_motion_3 = comp_motion_3.view((args.batch_size, args.num_segments) + comp_motion_3.size()[-3:])

            for batch_idx in range(args.batch_size):
                input_sample = input[batch_idx, :, :, :, :]
                tdn_motion_sample = tdn_motion[batch_idx, :, :, :, :]
                generated_motion_sample = generated_motion[batch_idx, :, :, :, :]

                comp_motion_sample_1 = comp_motion_1[batch_idx, :, :, :, :]
                comp_motion_sample_2 = comp_motion_2[batch_idx, :, :, :, :]
                comp_motion_sample_3 = comp_motion_3[batch_idx, :, :, :, :]

                video_folder = os.path.join(visualization_folder, str(i) + '_' + str(batch_idx))
                os.mkdir(video_folder)
                for channel in range(64):
                    tdn_channel_sample = tdn_motion_sample[:, channel, :, :]
                    generated_channel_sample = generated_motion_sample[:, channel, :, :]

                    generated_comp_sample_1 = comp_motion_sample_1[:, channel, :, :]
                    generated_comp_sample_2 = comp_motion_sample_2[:, channel, :, :]
                    generated_comp_sample_3 = comp_motion_sample_3[:, channel, :, :]

                    # plt.figure()
                    for t_idx in range(args.num_segments):
                        plt.subplot(6, args.num_segments, t_idx + 1)
                        plt.imshow(deprocess_image(input_sample[t_idx].permute(1, 2, 0).numpy()))
                        plt.axis('off')

                        plt.subplot(6, args.num_segments, args.num_segments + t_idx + 1)
                        plt.imshow(tdn_channel_sample[t_idx, :, :].cpu().numpy())
                        plt.axis('off')

                        plt.subplot(6, args.num_segments, args.num_segments * 2 + t_idx + 1)
                        plt.imshow(generated_channel_sample[t_idx, :, :].cpu().numpy())
                        plt.axis('off')

                        plt.subplot(6, args.num_segments, args.num_segments * 3 + t_idx + 1)
                        plt.imshow(generated_comp_sample_1[t_idx, :, :].cpu().numpy())
                        plt.axis('off')

                        plt.subplot(6, args.num_segments, args.num_segments * 4 + t_idx + 1)
                        plt.imshow(generated_comp_sample_2[t_idx, :, :].cpu().numpy())
                        plt.axis('off')

                        plt.subplot(6, args.num_segments, args.num_segments * 5 + t_idx + 1)
                        plt.imshow(generated_comp_sample_3[t_idx, :, :].cpu().numpy())
                        plt.axis('off')

                    fig_path = os.path.join(video_folder, 'channel ' + str(channel))
                    plt.savefig(fname=fig_path)
                    plt.cla()
                    plt.close('all')

        # loss = MSE_criterion(generated_motion, tdn_motion)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(('Visualization [{0}/{1}] done.\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         .format(i, len(loader), batch_time=batch_time, data_time=data_time)))  # TODO



if __name__ == '__main__':
    main()
