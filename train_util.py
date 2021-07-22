import os
import numpy as np
from os import listdir, path
from PIL import Image, ImageOps, ImageEnhance, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def sample_range(range, size=1, mean=False):
    if mean:
        return np.ones(size) * 0.5 * (range[1] - range[0]) + range[0]
    else:
        return np.random.rand(size) * (range[1] - range[0]) + range[0]


def perlin(sizes, rand=np.random.randint(5, 15), num_channel=1):
    img = np.zeros([sizes[0], sizes[1], num_channel])
    x, y = np.meshgrid(np.linspace(0, rand, sizes[1], endpoint=False),
                       np.linspace(0, rand, sizes[0], endpoint=False))
    xi = x.astype(int)
    yi = y.astype(int)
    xf = x - xi
    yf = y - yi
    u = 6 * xf ** 5 - 15 * xf ** 4 + 10 * xf ** 3
    v = 6 * yf ** 5 - 15 * yf ** 4 + 10 * yf ** 3
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    p = np.arange(rand * 3, dtype=int)
    p = np.stack([p, p]).flatten()
    for i in range(num_channel):
        np.random.shuffle(p)
        p_ = vectors[p[p[xi] + yi] % 4]
        n00 = p_[:, :, 0] * xf + p_[:, :, 1] * yf
        p_ = vectors[p[p[xi] + yi + 1] % 4]
        n01 = p_[:, :, 0] * xf + p_[:, :, 1] * (yf - 1)
        p_ = vectors[p[p[xi + 1] + yi + 1] % 4]
        n11 = p_[:, :, 0] * (xf - 1) + p_[:, :, 1] * (yf - 1)
        p_ = vectors[p[p[xi + 1] + yi] % 4]
        n10 = p_[:, :, 0] * (xf - 1) + p_[:, :, 1] * yf
        x1 = n00 + u * (n10 - n00)
        x2 = n01 + u * (n11 - n01)
        img[:, :, i] = x1 + v * (x2 - x1)

    img = np.clip(img.squeeze(), -0.5, 0.5) + 0.5
    return img


def fractal(sizes, num_channel=1, atte=0.5, octave=3, threshold=0):
    img = np.zeros((*sizes, num_channel)).squeeze()
    for i in range(octave):
        img += atte ** i * perlin(sizes, i * 8 + 5, num_channel)
    img = (img - img.min()) / (img.max() - img.min())
    if threshold > 0:
        img = (img > threshold).astype(float)
    return img


def fractal_save(num_images, sizes, dir, attn=(0.4, 0.8), octave=5):
    from tqdm import trange
    if not path.exists(dir):
        os.makedirs(dir)
    for i in trange(num_images):
        img = (fractal(sizes, atte=np.random.uniform(*attn), octave=octave) * 255).astype(np.uint8)
        Image.fromarray(img).save(path.join(dir, '{:05d}.png'.format(i)))


def fractal_image(size, scale=1):
    if isinstance(size, int):
        size = (size, size)
    path_img = fractal_imgs[int(np.random.rand() * len(fractal_imgs))]
    img = Image.open(path.join(dir_fractal, path_img))
    if scale != 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         resample=Image.BILINEAR)
    if img.width < size[1] or img.height < size[0]:
        img = ImageOps.expand(img, (0, 0, size[1] - img.width, size[0] - img.height))
    a = np.random.randint(0, img.width - size[1] + 1)
    b = np.random.randint(0, img.height - size[0] + 1)
    img = img.crop((a, b, a + size[1], b + size[0]))
    return img


def bg_image(size, grey=False, scale=1):
    if isinstance(size, int):
        size = (size, size)
    path_img = background_imgs[np.random.randint(len(background_imgs))]
    img = Image.open(path.join(dir_background, path_img))
    if scale != 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         resample=Image.BILINEAR)
    if img.width < size[1] or img.height < size[0]:
        img = ImageOps.expand(img, (0, 0, size[1] - img.width, size[0] - img.height))
    a = np.random.randint(0, img.width - size[1])
    b = np.random.randint(0, img.height - size[0])
    img = img.crop((a, b, a + size[1], b + size[0]))
    if grey:
        img = img.convert('L')
    return img


def img_patch(img, imsize, scale_type=Image.NEAREST):
    ratios = img.width * 1.0 / img.height, imsize[1] * 1.0 / imsize[0]
    if ratios[0] > ratios[1]:
        l = int(np.round(img.height * ratios[1]))
        a = (img.width - l) // 2
        img = img.resize(imsize,
                         resample=scale_type,
                         box=(a, 0, a + l, img.height))
    else:
        l = int(np.round(img.width / ratios[1]))
        a = (img.height - l) // 2
        img = img.resize(imsize,
                         resample=scale_type,
                         box=(0, a, img.width, a + l))
    return img


def obj_looks(img, a_hue=0, a_saturate=1, a_value=1, a_contrast=1, a_sharp=1, a_gamma=1):
    input_mode = img.mode
    if a_hue != 0 and input_mode not in ['L', '1', 'I', 'F']:
        h, s, v = img.convert('HSV').split()
        np_h = np.array(h, dtype=np.uint8)
        with np.errstate(over='ignore'):
            np_h += np.uint8(a_hue * 255)
        h = Image.fromarray(np_h, 'L')
        img = Image.merge('HSV', (h, s, v)).convert(input_mode)

    if a_saturate != 1:
        img = ImageEnhance.Color(img).enhance(a_saturate)

    if a_value != 1:
        img = ImageEnhance.Brightness(img).enhance(a_value)

    if a_contrast != 1:
        img = ImageEnhance.Contrast(img).enhance(a_contrast)

    if a_sharp != 1.:
        img = ImageEnhance.Sharpness(img).enhance(a_sharp)

    if a_gamma != 1:
        gain = 1
        input_mode = img.mode
        img = img.convert('RGB')
        gamma_map = [255 * gain * pow(ele / 255., a_gamma) for ele in range(256)] * 3
        img = img.point(gamma_map)
        img = img.convert(input_mode)
    return img


def depth_flip(img, bg):
    img = 255 - img
    img[bg] = 0
    return img


dir_background = 'dataset/background'
if not path.exists(dir_background):
    print('Downloading background images...')
    os.system('wget http://images.cocodataset.org/zips/val2017.zip')
    os.system('unzip val2017.zip')
    os.makedirs(dir_background)
    os.system('mv val2017/* {}'.format(dir_background))
    os.system('rm -r val2017 val2017.zip')
background_imgs = listdir(dir_background)
dir_fractal = 'dataset/fractal'
if not path.exists(dir_fractal):
    print('Generating fractal images...')
    os.makedirs(dir_fractal)
    fractal_save(4096, [128, 128], dir_fractal)
fractal_imgs = listdir(dir_fractal)
