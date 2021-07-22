import os
import sys
import yaml
import numpy as np
from PIL import Image
from tqdm import trange, tqdm
from collections import namedtuple

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data.dataloader as loader
import torch.nn.functional as F

from train_dataset import DataServoStereo
import train_model as model

# settings
arg = yaml.load(open(sys.argv[1], 'r'), yaml.Loader)
arg = namedtuple('Arg', arg.keys())(**arg)

# system init
cudnn.enabled = True
cudnn.benchmark = True
cudnn.deterministic = True
torch.manual_seed(0)
np.random.seed(0)

# model
kper = model.KeyPointGaussian(arg.sigma_kp[0], (arg.num_keypoint, *arg.im_size[1]))
enc = model.Encoder(arg.num_input, arg.num_keypoint, arg.growth_rate[0], arg.blk_cfg_enc, arg.drop_rate, kper).cuda()
dec = model.Decoder(arg.num_keypoint, arg.growth_rate[1], arg.blk_cfg_dec, arg.num_output).cuda()
cvt = model.ConverterServo(arg.num_keypoint * 2 * 3, arg.growth_rate[2], arg.blk_cfg_cvt, [sum(arg.motion_vec), 1]).cuda()

# optimizer
optim = torch.optim.Adam([{'params': enc.parameters(),
                           'weight_decay': arg.wd[0]},
                          {'params': dec.parameters(),
                           'weight_decay': arg.wd[1]},
                          {'params': cvt.parameters(),
                           'weight_decay': arg.wd[2]}],
                         lr=arg.lr, amsgrad=True)
print('enc parameters: {}'.format(sum([p.data.nelement() for p in enc.parameters()])))
print('dec parameters: {}'.format(sum([p.data.nelement() for p in dec.parameters()])))
print('cvt parameters: {}'.format(sum([p.data.nelement() for p in cvt.parameters()])))


def train(ep, loader_train):
    for i, (inL0, inR0,
            outDL, outDR, outSL, outSR,
            vecM, intV) in enumerate(loader_train):
        # data
        inL0 = inL0.cuda()
        inR0 = inR0.cuda()
        outDL = outDL.cuda()
        outDR = outDR.cuda()
        outSL = outSL.cuda()
        outSR = outSR.cuda()
        vecM = vecM.cuda()
        intV = intV.cuda()

        # lr scheduler update
        ith = ep * len(loader_train.dataset) // arg.batch_size + i, \
              arg.ep_train * len(loader_train.dataset) // arg.batch_size
        # lr scheduler update
        adjust_lr(*ith)
        # update kp sigma
        kper.sigma = min(2.0 * ith[0] / ith[1], 1) * (arg.sigma_kp[1] - arg.sigma_kp[0]) + arg.sigma_kp[0]

        # boots
        boot_size = max((arg.ep_train - ep) * 1.0 / arg.ep_train, arg.bootstrap)
        boot_recon = int(boot_size * arg.im_size[0][0] * arg.im_size[0][1])
        boot_equal = int(boot_size * arg.im_size[1][0] * arg.im_size[1][1] * arg.num_keypoint)
        boot_batch = int(boot_size * arg.batch_size)

        # reconstruction
        keypL0 = enc(inL0)
        keypR0 = enc(inR0)
        depthL, segL = dec(keypL0[1])
        depthR, segR = dec(keypR0[1])
        lossD = (F.smooth_l1_loss(depthL, outDL, reduction='none').view(arg.batch_size, -1).
                 topk(boot_recon, sorted=False)[0].mean() +
                 F.smooth_l1_loss(depthR, outDR, reduction='none').view(arg.batch_size, -1).
                 topk(boot_recon, sorted=False)[0].mean()) / 2
        lossS = (F.cross_entropy(segL, outSL, reduction='none').view(arg.batch_size, -1).
                 topk(boot_recon, sorted=False)[0].mean() +
                 F.cross_entropy(segR, outSR, reduction='none').view(arg.batch_size, -1).
                 topk(boot_recon, sorted=False)[0].mean()) / 2

        # motion
        vec, speed = cvt(torch.cat((keypL0[0], keypR0[0]), dim=1))
        lossM = F.cosine_similarity(vec, vecM).mul(-1).add(1).mul(intV).\
            topk(boot_batch, sorted=False)[0].mean()
        lossV = F.binary_cross_entropy_with_logits(speed, intV, reduction='none').\
            topk(boot_batch, sorted=False)[0].mean()

        # concentration
        lossC = None
        if arg.concentrate != 0:
            lossC = []
            for idx_i in range(0, arg.num_keypoint - 1):
                for idx_j in range(idx_i + 1, arg.num_keypoint):
                    distL = torch.norm(torch.cat(
                        ((keypL0[0][:, idx_i] - keypL0[0][:, idx_j]).unsqueeze(1),
                         (keypL0[0][:, idx_i + arg.num_keypoint] - keypL0[0][:, idx_j + arg.num_keypoint]).unsqueeze(1)),
                        dim=1), dim=1)
                    distR = torch.norm(torch.cat(
                        ((keypR0[0][:, idx_i] - keypR0[0][:, idx_j]).unsqueeze(1),
                         (keypR0[0][:, idx_i + arg.num_keypoint] - keypR0[0][:, idx_j + arg.num_keypoint]).unsqueeze(1)),
                        dim=1), dim=1)
                    lossC.append(distL.mul(arg.concentrate).exp().mul(keypL0[0][:, idx_i + 2 * arg.num_keypoint] *
                                                                      keypL0[0][:, idx_j + 2 * arg.num_keypoint]).mean())
                    lossC.append(distR.mul(arg.concentrate).exp().mul(keypR0[0][:, idx_i + 2 * arg.num_keypoint] *
                                                                      keypR0[0][:, idx_j + 2 * arg.num_keypoint]).mean())
            lossC = sum(lossC) / len(lossC)

        # inside
        lossI = None
        if arg.inside != 0:
            inoutL = outSL.eq(0).float()
            inoutL = F.interpolate(inoutL.unsqueeze(1), size=keypL0[1].size()[2:], align_corners=False, mode='bilinear')
            inoutR = outSR.eq(0).float()
            inoutR = F.interpolate(inoutR.unsqueeze(1), size=keypL0[1].size()[2:], align_corners=False, mode='bilinear')
            lossI = arg.inside * (inoutL.mul(keypL0[1]).mean() + inoutR.mul(keypR0[1]).mean()) / 2

        # updates
        optim.zero_grad()
        sum([l for l in [lossD, lossS, lossM, lossV, lossC, lossI] if l is not None]).backward()
        optim.step()

        # printing
        if i == 0:
            tqdm.write('ep: {}, DSMVCI: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.
                       format(ep,
                              lossD.item(), lossS.item(),
                              lossM.item(), lossV.item(),
                              lossC.item(), lossI.item()))


def adjust_lr(ep, ep_train, bn=True):
    if arg.lr_anne == 'step':
        a_lr = 0.4 ** ((ep > (0.3 * ep_train)) +
                       (ep > (0.6 * ep_train)) +
                       (ep > (0.9 * ep_train)))
    elif arg.lr_anne == 'cosine':
        a_lr = (np.cos(np.pi * ep / ep_train) + 1) / 2
    elif arg.lr_anne == 'repeat':
        partition = [0, 0.15, 0.30, 0.45, 0.6, 0.8, 1.0]
        par = int(np.digitize(ep * 1. / ep_train, partition))
        T = (partition[par] - partition[par - 1]) * ep_train
        t = ep - partition[par - 1] * ep_train
        a_lr = 0.5 * (1 + np.cos(np.pi * t / T))
        a_lr *= 1 - partition[par - 1]
    else:
        a_lr = 1

    for param_group in optim.param_groups:
        param_group['lr'] = max(a_lr, 0.01) * arg.lr

    if bn:
        def fn(m):
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                m.momentum = min(max(a_lr, 0.01), 0.9)
        enc.apply(fn)
        dec.apply(fn)
        cvt.apply(fn)


def save_checkpoint(base_dir):
    state = {'enc_state_dict': enc.state_dict(),
             'dec_state_dict': dec.state_dict(),
             'cvt_state_dict': cvt.state_dict()}
    torch.save(state, os.path.join(base_dir, 'ckpt.pth'))
    print('checkpoint saved.')


def load_checkpoint(base_dir):
    cp_net = torch.load(os.path.join(base_dir, 'ckpt.pth'))
    enc.load_state_dict(cp_net['enc_state_dict'])
    dec.load_state_dict(cp_net['dec_state_dict'])
    cvt.load_state_dict(cp_net['cvt_state_dict'])
    print('checkpoint loaded.')


def test(loader_test):
    from skimage import transform
    color = yaml.load(open('cfg/color.yaml', 'r'), Loader=yaml.Loader)

    num_obj = len(set(arg.obj_class))
    sims, speeds = [], []
    for i, (inL, inR, _, _, _, _, vecM, intV) in enumerate(loader_test):
        inL = inL.cuda()
        inR = inR.cuda()
        vecM = vecM.cuda()
        intV = intV.cuda()

        # forward-pass
        keypL = enc(inL)
        keypR = enc(inR)
        depth, seg = dec(keypL[1])
        vec, speed = cvt(torch.cat((keypL[0], keypR[0]), dim=1))
        vec = F.cosine_similarity(vec, vecM.squeeze(), dim=0).mul(-1).add(1).mul(intV.squeeze()).detach().cpu().item()
        speed = F.binary_cross_entropy_with_logits(speed.unsqueeze(0), intV).squeeze().detach().cpu().item()

        # visual
        keyp = keypL[1].detach().squeeze().cpu().numpy()
        keyps = np.zeros((inL.size(2), inL.size(3), 3), np.float)
        for j in range(keyp.shape[0]):
            keyps = keyps + np.tile(transform.resize(keyp[j], keyps.shape[:2])[:, :, np.newaxis], [1, 1, 3]) * \
                    np.array(color[j]).reshape(1, 1, 3)
        keyps = (keyps * 255).round().astype(np.uint8)
        img = ((inL.detach().squeeze().cpu().numpy() * 0.25) * 255 + 128).round().clip(0, 255).astype(np.uint8)
        depth = ((depth.detach().squeeze().cpu().numpy() * 0.25) * 255 + 128).round().clip(0, 255).astype(np.uint8)
        seg = (seg.squeeze().argmax(dim=0).detach().cpu().numpy() * 255. / (num_obj - 1)).astype(np.uint8)
        Image.fromarray(np.hstack((np.tile(img[:, :, None], [1, 1, 3]), keyps,
                                   np.tile(depth[:, :, None], [1, 1, 3]), np.tile(seg[:, :, None], [1, 1, 3])))). \
            save(os.path.join(arg.dir_base, 'test/{:04d}_{:.2f}_{:.2f}.png'.format(i, vec, speed)))
        sims.append(vec)
        speeds.append(speed)
    print('Average vector loss: ', sum(sims) / len(sims))
    print('Average speed loss : ', sum(speeds) / len(speeds))


def main():
    if arg.task in ['full']:
        # data directory
        if not os.path.exists(arg.dir_base):
            os.makedirs(arg.dir_base)
        os.system('cp {} {}'.format(sys.argv[1], os.path.join(arg.dir_base, 'servo.yaml')))

        # load database
        ds_train = DataServoStereo(arg)
        data_param = {'pin_memory': False, 'shuffle': True, 'batch_size': arg.batch_size, 'drop_last': True,
                      'num_workers': 8, 'worker_init_fn': lambda _: np.random.seed(ord(os.urandom(1)))}
        loader_train = loader.DataLoader(ds_train, **data_param)

        # training
        enc.train()
        dec.train()
        cvt.train()
        print('training...')
        for ep in trange(arg.ep_train):
            train(ep, loader_train)

        # save
        save_checkpoint(arg.dir_base)

    if arg.task in ['full', 'test']:
        # directory
        if not os.path.isdir(os.path.join(arg.dir_base, 'test')):
            os.makedirs(os.path.join(arg.dir_base, 'test'))

        # dataset
        ds_test = DataServoStereo(arg, False)
        loader_test = loader.DataLoader(ds_test)

        # load model
        load_checkpoint(arg.dir_base)
        enc.eval()
        dec.eval()
        cvt.eval()
        kper.sigma = arg.sigma_kp[1]

        # test
        print('testing...')
        test(loader_test)


if __name__ == '__main__':
    main()
