import os
from time import time

import argparse
import torch
import math
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datetime import datetime as dt

from net.U_Net3D import unet_3D
from loss.mse_loss import MSELoss
from load_data import DSAReconDataset

if __name__ == '__main__':
    # train parameter
    parser = argparse.ArgumentParser(description='DSA 3D Reconstruction Training')
    parser.add_argument('--train_input_path', type=str,
                        help='2d input image and 2d label')
    parser.add_argument('--result_path', type=str,
                        help='training output path that save logs and ckpt')
    parser.add_argument('--epoch', type=int, default=1500,
                        help='the number of epoch')
    parser.add_argument('--leaing_rate', type=float, default=1e-3,
                        help='the value of leaing_rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='the value of weight_decay')
    parser.add_argument('--optim', type=str, default='adam',
                        help='network optimizer')
    parser.add_argument('--train_batch_size', type=int, default=3,
                        help='batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader')
    parser.add_argument('--stage', type=int, default=1,
                        help='the number of stage of reconstruction network')
    parser.add_argument('--last_stage_path', type=str, default=None,
                        help='the path where the output of the previous/last stage of the network is saved')

    # model parameter
    parser.add_argument("--view_num", type=int, default=6,
                        help='number of views for the image input')
    parser.add_argument("--loss", type=str, default='MSE',
                        help='loss function')
    parser.add_argument("--check_point_save", type=int, default=100,
                        help='epoch interval to save the model.pth')
    parser.add_argument("--pretrain_model_path", type=str, default=None,
                        help='path of pretrain model')

    args = parser.parse_args()

    # parameter
    cudnn.benchmark = True
    Epoch = args.epoch
    leaing_rate = args.leaing_rate
    views = args.view_num
    perangle = 180/views
    batch_size = args.train_batch_size
    num_workers = args.num_workers
    pin_memory = False
    stage = args.stage

    # Network
    if stage == 1:
        net = unet_3D(in_channels=views)
    else:
        net = unet_3D(in_channels=views+1)
    net = net.cuda()
    # net = torch.nn.DataParallel(net.cuda(),device_ids=range(torch.cuda.device_count()))
    if args.pretrain_model_path is not None:
        net.load_state_dict(torch.load(args.pretrain_model_path))

    # Dataloader
    if stage == 1:
        train_dataset = DSAReconDataset(stage, views, args.train_input_path)
    else:
        train_dataset = DSAReconDataset(stage, views, args.train_input_path, args.last_stage_path)
    train_dl = DataLoader(train_dataset, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)

    # Loss function
    if args.loss == 'MSE':
        loss_func = MSELoss(views)
    else:
        assert False, print('Not implemented loss: {}'.format(args.loss))

    # optimizer
    if args.optim == 'adam':
        opt = torch.optim.Adam(net.parameters(), lr=leaing_rate, weight_decay=args.weight_decay)
    else:
        assert False, print('Not implemented optimizer: {}'.format(args.optim))

    # log
    result_path = args.result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(os.path.join(result_path, 'logs')):
        os.mkdir(os.path.join(result_path, 'logs'))
    if not os.path.exists(os.path.join(result_path, 'checkpoints')):
        os.mkdir(os.path.join(result_path, 'checkpoints'))
    log_dir = os.path.join(result_path, 'logs', dt.now().isoformat())
    ckpt_dir = os.path.join(result_path, 'checkpoints', dt.now().isoformat())
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    start = time()

    # Training loop
    for epoch in range(Epoch):
        mean_loss = []

        for step, (ct, seg) in enumerate(train_dl):
            ct = ct.cuda()
            seg = seg.cuda()

            outputs = net(ct)
            outputs = outputs.squeeze(1)
            loss, pred_proj = loss_func(outputs, seg)

            mean_loss.append(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_ = leaing_rate * (1.0 - (epoch*math.floor(50/batch_size)+step) / Epoch/math.floor(50/batch_size)) ** 0.9 
            for param_group in opt.param_groups:
                param_group['lr'] = lr_

            if step % 4 == 0:
                print('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
                    .format(epoch, step, loss.item(), (time() - start) / 60))
            n_itr = epoch * math.floor(50/batch_size) + step
            train_writer.add_scalar('TrainBatchLoss', loss.item(), n_itr)

        mean_loss = sum(mean_loss) / len(mean_loss)
        train_writer.add_scalar('TrainEpochLoss', mean_loss, epoch)
        # print('TrainEpochLoss:',epoch, mean_loss)

        # save checkpoint
        cptn = args.check_point_save
        if (epoch+1) % cptn == 0 or (epoch == Epoch-2):
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            torch.save(net.state_dict(), ckpt_dir + '/UNet{}-{:.3f}-{:.3f}.pth'.format(epoch, loss.item(), mean_loss))

        # save visualization output
        if (epoch+1) % cptn == 0 or epoch == 0 :
            index = 0
            # check label project
            label = seg[index,:,:,:].cpu()
            label = label.detach().numpy()
            for i in range(views):
                train_writer.add_image('Train Sample#%02d_%02dview/Volume GroundTruth' % (index, i), np.expand_dims(label[i,:,:],axis=0), epoch)

            # check predict project
            pred_proj_np = pred_proj[index,:,:,:].cpu()
            pred_proj_np = pred_proj_np.detach().numpy()
            for i in range(views):
                train_writer.add_image('Train Sample#%02d_%02dview/Volume predict' % (index, i), np.expand_dims(pred_proj_np[i,:,:],axis=0), epoch)

            # # check last project
            # if stage>1:
            #     inputt = ct[index,:,:,:,:].cpu() 
            #     inputt = inputt.detach().numpy()
            #     for i in range(views):
            #         inputt22 = oblique_project1(inputt[views,:,:,:], perangle*i + perangle)
            #         plt.imshow(inputt22, cmap="gray")
            #         plt.savefig(result_path + '/input_last_project_'+str(i)+'.jpg')