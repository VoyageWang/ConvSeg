import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import tifffile as tiff
import matplotlib.patches as mpatches
from PIL import Image
import random
from .transform import *

CLASSES = ('water', 'land')
PALETTE = [[0, 0, 255], [255, 0, 0]]

ORIGIN_IMG_SIZE = (256, 256)
INPUT_IMG_SIZE = (256, 256)
TEST_IMG_SIZE = (256, 256)


# def standardization(data):
#     mu = np.mean(data, axis=0)
#     sigma = np.std(data, axis=0)
#     return (data - mu) / sigma


## 影像拉伸 ##
def img_processing(im_band, img_data):
    if im_band == 1:
        data_jpg = np.zeros((img_data.shape[0], img_data.shape[1]), dtype='uint8')
        im_max = np.amax(img_data)
        im_min = np.amin(img_data)
        for m in range(0, img_data.shape[0]):
            for n in range(0, img_data.shape[1]):
                data_jpg[m,n] = float(255./(im_max-im_min)) * (img_data[m,n] - im_min)
    else:
        data_jpg = np.zeros((img_data.shape[1],img_data.shape[2],3),dtype='uint8')
        for i in range(3):
            im_max = np.amax(img_data[i,:,:])
            im_min = np.amin(img_data[i,:,:])
            for m in range(0, img_data.shape[1]):
                for n in range(0, img_data.shape[2]):
                    data_jpg[m,n,i] = float(255./(im_max-im_min))*(img_data[i,m,n]-im_min)
    return data_jpg


## GAMMA 变换 ##
def gamma_trans(img, gamma):
    gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)


def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def train_aug(img, mask):
    # crop_aug = Compose([RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode='value'),
    #                     SmartCropV1(crop_size=256, max_ratio=0.75,
    #                                 ignore_index=len(CLASSES), nopad=False)])
    # img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    # img = np.transpose(img, (2, 0 ,1))
    # img = img_processing(12, img)
    # img = gamma_trans(img, 0.5)
    # img = img.reshape((12, 256, 256))
    mask = mask.reshape((256, 256))
    mask[mask < 0] = 0
    # aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    # img, mask = aug['image'], aug['mask']
    # cv2.normalize(img, dst=img)
    # img = standardization(img)
    return img, mask


def get_training_transform():
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    img.dtype = np.int16
    mask.dtype = np.int16
    # img = img_processing(12, img)
    # img = gamma_trans(img, 0.5)
    mask[mask < 0] = 0
    mask = mask.reshape((256, 256))
    # aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    # img = standardization(img)
    # print(img.shape)
    # cv2.normalize(img, dst=img)
    # img, mask = aug['image'], aug['mask']
    return img, mask


class SWEDDataset(Dataset):
    def __init__(self, data_root='/data2/wangyuji/Geoseg/SWED/test', mode='val', img_dir='images', mask_dir='labels',
                 train_suffix='.npy', test_suffix='.tif', transform=val_aug, mosaic_ratio=0.0,
                 img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.train_suffix = train_suffix
        self.test_suffix = test_suffix
        self.transform = transform
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)

    def __getitem__(self, index):
        p_ratio = random.random()
        if self.mode == 'val' or self.mode == 'test':
            # print('t')
            img, mask = self.load_img_and_mask_test(index)
            if self.transform:
                img, mask = self.transform(img, mask)
            img = torch.from_numpy(img).permute(0, 1, 2).float()
            # mask = mask/1.0
            mask = torch.from_numpy(mask).long()
            img_id = self.img_ids[index]
            results = dict(img_id=img_id, img=img, gt_semantic_seg=mask)
            return results
        elif self.mode == 'train':
            # print('g')
            # img, mask = self.load_mosaic_img_and_mask(index)
            img, mask = self.load_img_and_mask_train(index)
            if self.transform:
                img, mask = self.transform(img, mask)
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            # mask = mask/1.0
            mask = torch.from_numpy(mask).long()
            img_id = self.img_ids[index]
            results = dict(img_id=img_id, img=img, gt_semantic_seg=mask)
            return results
        # img = torch.from_numpy(img).permute(0, 1 ,2).float()
        # mask = torch.from_numpy(mask).long()
        # img_id = self.img_ids[index]
        # results = dict(img_id=img_id, img=img, gt_semantic_seg=mask)
        # return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        print(data_root)
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        print(len(img_filename_list))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        print(len(mask_filename_list))
        assert len(img_filename_list) == len(mask_filename_list)
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids

    def load_img_and_mask_train(self, index):
        img_id = self.img_ids[index]
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.train_suffix)
        img_id = img_id.replace('_chip_', '_image_')
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.train_suffix)
        # img = Image.open(img_name).convert('RGB')
        # mask = Image.open(mask_name).convert('L')
        img = np.load(img_name)
        mask = np.load(mask_name)
        return img, mask
    
    def load_img_and_mask_test(self, index):
        img_id = self.img_ids[index]
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.test_suffix)
        # print(img_id)
        img_id = img_id.replace('_label_', '_image_')
        # print(img_id)
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.test_suffix)
        img = tiff.imread(img_name)
        mask = tiff.imread(mask_name)
        # img = Image.open(img_name).convert('RGB')
        # mask = Image.open(mask_name).convert('L')
        # img = np.load(img_name)
        # mask = np.load(mask_name)
        return img, mask

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)

        h = self.img_size[0]
        w = self.img_size[1]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        # print(img.shape)

        return img, mask


def show_img_mask_seg(seg_path, img_path, mask_path, start_seg_index):
    seg_list = os.listdir(seg_path)
    seg_list = [f for f in seg_list if f.endswith('.png')]
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    seg_list = seg_list[start_seg_index:start_seg_index+2]
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f'{seg_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert('P')
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert('RGB'))
        mask = cv2.imread(f'{mask_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask).convert('P')
        mask.putpalette(np.array(PALETTE, dtype=np.uint8))
        mask = np.array(mask.convert('RGB'))
        img_id = str(seg_id.split('.')[0])+'.tif'
        img = cv2.imread(f'{img_path}/{img_id}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, 0].set_axis_off()
        ax[i, 0].imshow(img)
        ax[i, 0].set_title('RS IMAGE ' + img_id)
        ax[i, 1].set_axis_off()
        ax[i, 1].imshow(mask)
        ax[i, 1].set_title('Mask True ' + seg_id)
        ax[i, 2].set_axis_off()
        ax[i, 2].imshow(img_seg)
        ax[i, 2].set_title('Mask Predict ' + seg_id)
        ax[i, 2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


def show_seg(seg_path, img_path, start_seg_index):
    seg_list = os.listdir(seg_path)
    seg_list = [f for f in seg_list if f.endswith('.png')]
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    seg_list = seg_list[start_seg_index:start_seg_index+2]
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f'{seg_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert('P')
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert('RGB'))
        img_id = str(seg_id.split('.')[0])+'.tif'
        img = cv2.imread(f'{img_path}/{img_id}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, 0].set_axis_off()
        ax[i, 0].imshow(img)
        ax[i, 0].set_title('RS IMAGE '+img_id)
        ax[i, 1].set_axis_off()
        ax[i, 1].imshow(img_seg)
        ax[i, 1].set_title('Seg IMAGE '+seg_id)
        ax[i, 1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


def show_mask(img, mask, img_id):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(np.array(PALETTE, dtype=np.uint8))
    mask = np.array(mask.convert('RGB'))
    ax1.imshow(img)
    ax1.set_title('RS IMAGE ' + str(img_id)+'.tif')
    ax2.imshow(mask)
    ax2.set_title('Mask ' + str(img_id)+'.png')
    ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')
