import yaml
import torch
import numpy as np
from os import listdir, path
from PIL import Image, ImageFilter

import torch.utils.data as _data
import torchvision.transforms.functional as F
from skimage.segmentation import find_boundaries
from train_util import sample_range, img_patch, obj_looks, bg_image, fractal_image, depth_flip


class DataServoStereo(_data.Dataset):
    def __init__(self, arg, train=True):
        self.train = train
        self.im_size = arg.im_size[0]
        self.obj_class = arg.obj_class
        self.texture = arg.texture
        self.hue = arg.hue
        self.saturation = arg.saturation
        self.brightness = arg.brightness
        self.contrast = arg.contrast
        self.sharp = arg.sharp
        self.gamma = arg.gamma
        self.blur = arg.blur
        self.noise = arg.noise
        self.bg = arg.bg
        self.mean = arg.mean
        self.std = arg.std
        self.motion_vec = arg.motion_vec
        self.a_dist = arg.a_dist

        self.data_path = path.join(arg.dir_dataset, 'train' if train else 'test')
        self.ims = sorted(listdir(path.join(self.data_path, 'left/color')))
        gt_yml = path.join(self.data_path, 'gt.yaml')
        self.gts = yaml.load(open(gt_yml, 'r'), yaml.CLoader)

        print('{} {} data loaded'.format(len(self.ims), 'training' if train else 'testing'))

    def __getitem__(self, index):
        return self.data_transform(index)

    def __len__(self):
        return len(self.ims)

    def data_transform(self, index):
        # images
        inL0, outDL, outSL = img_proc_servo(path.join(self.data_path, 'left'), self.ims[index],
                                            self.im_size, self.obj_class, self.texture,
                                            self.hue, self.saturation, self.brightness,
                                            self.contrast, self.sharp, self.gamma,
                                            self.blur, self.noise, self.bg, self.mean, self.std)
        inR0, outDR, outSR = img_proc_servo(path.join(self.data_path, 'right'), self.ims[index],
                                            self.im_size, self.obj_class, self.texture,
                                            self.hue, self.saturation, self.brightness,
                                            self.contrast, self.sharp, self.gamma,
                                            self.blur, self.noise, self.bg, self.mean, self.std)

        # label
        idx = self.ims[index].split('.')[0].split('_')
        gt = self.gts[int(idx[0])]
        vecM = torch.FloatTensor(np.array(gt[:-1])[self.motion_vec])
        intV = 1 - np.exp(float(idx[1]) * gt[-1] * self.a_dist)

        return inL0, inR0, outDL, outDR, outSL, outSR, vecM, intV


def img_proc_servo(data_path, im, im_size, obj_class,
                   texture, hue, saturation, brightness, contrast, sharp, gamma,
                   blur, noise, bg, mean, std):
    # output segmentation
    outS = Image.open(path.join(data_path, 'segme', im))
    outS = img_patch(outS, im_size)
    outS = np.array(outS)
    for i, v in enumerate(obj_class):
        if i != v:
            outS[outS == i] = v
    idx_bg = outS == 0

    outS = torch.from_numpy(outS.astype(np.int))

    # output depth
    outD = Image.open(path.join(data_path, 'depth', im))
    outD = img_patch(outD, im_size)
    outD = depth_flip(np.array(outD), idx_bg)
    outD = F.to_tensor(outD)
    outD = F.normalize(outD, [0.5], [0.25])

    # input color
    inC0 = img_proc(path.join(data_path, 'color', im), idx_bg, im_size,
                    texture, bg,
                    hue, saturation, brightness, contrast, sharp, gamma,
                    blur, noise, mean, std, True)

    return inC0, outD, outS


def img_proc(data_path, idx_bg, im_size,
             texture, bg,
             hue, saturation, brightness, contrast, sharp, gamma,
             blur, noise, mean, std, grey):
    inC = Image.open(data_path)
    inC = img_patch(inC, im_size)
    if grey:
        inC = inC.convert('L')
    if sum(texture) > 0:
        inC.paste(bg_image(im_size, grey),
                  mask=Image.fromarray(np.bitwise_not(idx_bg) * np.uint8(np.random.uniform(*texture))))
    if np.random.rand() < bg:
        inC.paste(bg_image(im_size, grey),
                  mask=Image.fromarray(idx_bg * np.uint8(255)))
    else:
        inC.paste(Image.new('L' if grey else 'RGB', im_size, int(255 * mean)),
                  mask=Image.fromarray(idx_bg * np.uint8(255)))
    inC = obj_looks(inC,
                    a_hue=sample_range(hue), a_saturate=sample_range(saturation), a_value=sample_range(brightness),
                    a_contrast=sample_range(contrast), a_sharp=sample_range(sharp), a_gamma=sample_range(gamma))
    if sum(blur) > 0:
        edge_blur = np.random.uniform(*blur)
        inC.paste(inC.filter(ImageFilter.GaussianBlur(edge_blur)),
                  mask=Image.fromarray(find_boundaries(idx_bg) * np.uint8(255)).
                  filter(ImageFilter.GaussianBlur(edge_blur)))
    inC = np.array(inC, np.uint8)
    if sum(noise) > 0:
        fractal = (np.array(fractal_image(im_size)) - 127.) / 128.
        inC = (inC + (fractal if grey else fractal[:, :, None]) * np.random.uniform(*noise)). \
            clip(0, 255).round().astype(np.uint8)

    inC = F.to_tensor(inC)
    inC = F.normalize(inC, [mean], [std])

    return inC
