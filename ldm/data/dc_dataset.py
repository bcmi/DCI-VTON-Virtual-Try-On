import random

import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import numpy as np
import json
from typing import List, Tuple
from data.labelmap import label_map
from numpy.linalg import lstsq


def show(title, array):
    plt.title(title)
    plt.imshow(array)
    plt.show()


def mask2bbox(mask):
    up = np.max(np.where(mask)[0])
    down = np.min(np.where(mask)[0])
    left = np.min(np.where(mask)[1])
    right = np.max(np.where(mask)[1])
    center = ((up + down) // 2, (left + right) // 2)

    factor = random.random() * 0.1 + 0.1

    up = int(min(up * (1 + factor) - center[0] * factor + 1, mask.shape[0]))
    down = int(max(down * (1 + factor) - center[0] * factor, 0))
    left = int(max(left * (1 + factor) - center[1] * factor, 0))
    right = int(min(right * (1 + factor) - center[1] * factor + 1, mask.shape[1]))
    return down, up, left, right


def get_agnostic(parse_array, pose_data, category, size):
    parse_shape = (parse_array > 0).astype(np.float32)

    parse_head = (parse_array == 1).astype(np.float32) + \
                 (parse_array == 2).astype(np.float32) + \
                 (parse_array == 3).astype(np.float32) + \
                 (parse_array == 11).astype(np.float32)

    parser_mask_fixed = (parse_array == label_map["hair"]).astype(np.float32) + \
                        (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["hat"]).astype(np.float32) + \
                        (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                        (parse_array == label_map["scarf"]).astype(np.float32) + \
                        (parse_array == label_map["bag"]).astype(np.float32)

    parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

    arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)

    if category == 'dresses':
        label_cat = 7
        parse_mask = (parse_array == 7).astype(np.float32) + \
                     (parse_array == 12).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    elif category == 'upper_body':
        label_cat = 4
        parse_mask = (parse_array == 4).astype(np.float32)

        parser_mask_fixed += (parse_array == label_map["skirt"]).astype(np.float32) + \
                             (parse_array == label_map["pants"]).astype(np.float32)

        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    elif category == 'lower_body':
        label_cat = 6
        parse_mask = (parse_array == 6).astype(np.float32) + \
                     (parse_array == 12).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32)

        parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                             (parse_array == 14).astype(np.float32) + \
                             (parse_array == 15).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    parse_head = torch.from_numpy(parse_head)  # [0,1]
    parse_mask = torch.from_numpy(parse_mask)  # [0,1]
    parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
    parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

    # dilation
    parse_without_cloth = np.logical_and(parse_shape, np.logical_not(parse_mask))
    parse_mask = parse_mask.cpu().numpy()

    width = size[0]
    height = size[1]

    im_arms = Image.new('L', (width, height))
    arms_draw = ImageDraw.Draw(im_arms)
    if category == 'dresses' or category == 'upper_body':
        shoulder_right = tuple(np.multiply(pose_data[2, :2], height / 512.0))
        shoulder_left = tuple(np.multiply(pose_data[5, :2], height / 512.0))
        elbow_right = tuple(np.multiply(pose_data[3, :2], height / 512.0))
        elbow_left = tuple(np.multiply(pose_data[6, :2], height / 512.0))
        wrist_right = tuple(np.multiply(pose_data[4, :2], height / 512.0))
        wrist_left = tuple(np.multiply(pose_data[7, :2], height / 512.0))
        if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
            if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                arms_draw.line([wrist_left, elbow_left, shoulder_left, shoulder_right], 'white', 30, 'curve')
            else:
                arms_draw.line([wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right], 'white', 30,
                               'curve')
        elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
            if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                arms_draw.line([shoulder_left, shoulder_right, elbow_right, wrist_right], 'white', 30, 'curve')
            else:
                arms_draw.line([elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right], 'white', 30,
                               'curve')
        else:
            arms_draw.line([wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right], 'white',
                           30, 'curve')

        if height > 512:
            im_arms = cv2.dilate(np.float32(im_arms), np.ones((10, 10), np.uint16), iterations=5)
        elif height > 256:
            im_arms = cv2.dilate(np.float32(im_arms), np.ones((5, 5), np.uint16), iterations=5)
        hands = np.logical_and(np.logical_not(im_arms), arms)
        parse_mask += im_arms
        parser_mask_fixed += hands

    # delete neck
    parse_head_2 = torch.clone(parse_head)
    if category == 'dresses' or category == 'upper_body':
        points = []
        points.append(np.multiply(pose_data[2, :2], height / 512.0))
        points.append(np.multiply(pose_data[5, :2], height / 512.0))
        x_coords, y_coords = zip(*points)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords, rcond=None)[0]
        for i in range(parse_array.shape[1]):
            y = i * m + c
            parse_head_2[int(y - 20 * (height / 512.0)):, i] = 0

    parser_mask_fixed = np.logical_or(parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
    parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                           np.logical_not(np.array(parse_head_2, dtype=np.uint16))))

    if height > 512:
        parse_mask = cv2.dilate(parse_mask, np.ones((20, 20), np.uint16), iterations=5)
    elif height > 256:
        parse_mask = cv2.dilate(parse_mask, np.ones((10, 10), np.uint16), iterations=5)
    else:
        parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
    parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
    agnostic_mask = parse_mask_total.unsqueeze(0)
    return agnostic_mask


class Dataset(data.Dataset):
    def __init__(self, dataroot_path: str,
                 phase: str,
                 order: str = 'paired',
                 category: str = 'all',
                 size: int = 512):
        """
        Initialize the PyTorch Dataset Class
        :param dataroot_path: dataset root folder
        :type dataroot_path:  string
        :param phase: phase (train | test)
        :type phase: string
        :param order: setting (paired | unpaired)
        :type order: string
        :param category: clothing category (all | upper_body | lower_body | dresses)
        :type category: str
        :param size: image size (height, width)
        :type size: int
        """
        super(Dataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.category = ['dresses', 'upper_body', 'lower_body'] if category == 'all' else [category]
        self.height = size
        self.width = size
        self.load_size =(int(size / 256 * 192), size)
        self.radius = 5
        self.to_tensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.to_tensor = transforms.ToTensor()
        self.clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                   (0.26862954, 0.26130258, 0.27577711))

        im_names = []
        c_names = []
        dataroot_names = []

        for c in self.category:
            assert c in ['dresses', 'upper_body', 'lower_body']

            dataroot = os.path.join(self.dataroot, c)
            if phase == 'train':
                filename = os.path.join(dataroot, f"{phase}_pairs.txt")
            else:
                filename = os.path.join(dataroot, f"{phase}_pairs_{order}.txt")
            with open(filename, 'r') as f:
                for line in f.readlines():
                    im_name, c_name = line.strip().split()
                    im_names.append(im_name)
                    c_names.append(c_name)
                    dataroot_names.append(dataroot)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names

    def __getitem__(self, index):
        """
        For each index return the corresponding sample in the dataset
        :param index: data index
        :type index: int
        :return: dict containing dataset samples
        :rtype: dict
        """
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        dataroot = self.dataroot_names[index]

        # Clothing image
        cloth = Image.open(os.path.join(dataroot, 'images', c_name))
        cloth = cloth.resize((self.width, self.height))
        cloth = self.transform(cloth)  # [-1,1]

        # Clothing mask
        cloth_mask = Image.open(os.path.join(dataroot, 'masks', c_name.replace('.jpg', '.png')))
        cloth_mask = cloth_mask.resize((self.width, self.height))
        cloth_mask = self.to_tensor(cloth_mask)  # [0,1]
        cloth_mask = (cloth_mask > 0.5).float()

        down, up, left, right = mask2bbox(cloth_mask[0].numpy())
        ref_image = cloth[:, down:up, left:right]
        ref_image = (ref_image + 1.0) / 2.0
        show('ref_image', ref_image.permute(1, 2, 0))
        ref_image = transforms.Resize((224, 224))(ref_image)
        ref_image = self.clip_normalize(ref_image)

        # Person image
        im = Image.open(os.path.join(dataroot, 'images', im_name))
        im = im.resize((self.width, self.height))
        im = self.transform(im)  # [-1,1]

        warped_cloth = Image.open(os.path.join(dataroot, 'warped_cloth', im_name))
        warped_cloth = warped_cloth.resize((self.width, self.height))
        warped_cloth = self.transform(warped_cloth)

        warped_mask = Image.open(os.path.join(dataroot, 'warped_mask', im_name.replace('.jpg', '.png')))
        warped_mask = warped_mask.resize((self.width, self.height))
        warped_mask = self.to_tensor(warped_mask)

        warped_cloth = warped_cloth * warped_mask

        # Skeleton
        skeleton = Image.open(os.path.join(dataroot, 'skeletons', im_name.replace("_0", "_5")))
        skeleton = skeleton.resize((self.width, self.height))
        skeleton = self.transform(skeleton)

        # Label Map
        parse_name = im_name.replace('_0.jpg', '_4.png')
        im_parse = Image.open(os.path.join(dataroot, 'label_maps', parse_name))
        im_parse = im_parse.resize(self.load_size, Image.NEAREST)
        parse_array = np.array(im_parse)

        # Load pose points
        pose_name = im_name.replace('_0.jpg', '_2.json')
        with open(os.path.join(dataroot, 'keypoints', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 4))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.height, self.width)
        r = self.radius * (self.height / 512.0)
        for i in range(point_num):
            one_map = Image.new('L', (self.width, self.height))
            draw = ImageDraw.Draw(one_map)
            point_x = np.multiply(pose_data[i, 0], self.width / 384.0)
            point_y = np.multiply(pose_data[i, 1], self.height / 512.0)
            if point_x > 1 and point_y > 1:
                draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
            one_map = self.to_tensor(one_map)
            pose_map[i] = one_map[0]

        agnostic_mask = get_agnostic(parse_array, pose_data, dataroot.split('/')[-1], self.load_size)
        agnostic_mask = transforms.functional.resize(agnostic_mask, (self.height, self.width),
                                                     interpolation=transforms.InterpolationMode.NEAREST)

        inpaint_mask = 1 - agnostic_mask
        inpaint = im * agnostic_mask + warped_cloth * inpaint_mask
        feat = inpaint

        # uv = np.load(os.path.join(dataroot, 'dense', im_name.replace('_0.jpg', '_5_uv.npz')))
        # uv = uv['uv']
        # uv = torch.from_numpy(uv)
        # uv = transforms.functional.resize(uv, (self.height, self.width))
        #
        # labels = Image.open(os.path.join(dataroot, 'dense', im_name.replace('_0.jpg', '_5.png')))
        # labels = labels.resize((self.width, self.height), Image.NEAREST)
        # labels = torch.from_numpy(np.array(labels)[None]).long()
        # dense_labels = torch.FloatTensor(25, self.height, self.width).zero_()
        # dense_labels = dense_labels.scatter_(0, labels, 1.0)

        show('inpaint_mask', inpaint_mask[0])
        show('im', (im.permute(1, 2, 0) + 1) / 2)
        show('inpaint', (inpaint.permute(1, 2, 0) + 1) / 2)
        result = {
            'file_name': im_name,  # for visualization or ground truth
            "GT": im,
            "inpaint_image": inpaint,
            "inpaint_mask": inpaint_mask,
            "ref_imgs": ref_image,
            'warp_feat': feat
        }

        return result

    def __len__(self):
        return len(self.c_names)
