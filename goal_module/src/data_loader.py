# -*- coding: utf-8 -*-
import os
import pickle
import random
import albumentations as A

import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
import numpy as np
import cv2
from PIL import Image, ImageDraw

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from src.models.model_utils.cnn_big_images_utils import create_tensor_image
from src.models.model_utils.visual_prompt import draw_arrow, draw_dotted


# +
class dataset_set_name(BaseDataset):
    """
    Dataset class to load iteratively pre-made batches saved as pickle files.
    Apply data augmentation when needed.
    """

    def __init__(self, args, set_name):

        self.path_to_folder = os.path.join(
            args.save_dir, "data_batches", f"{set_name}_batches")

        self.ids = sorted(os.listdir(self.path_to_folder))
        self.args = args
        self.data_augmentation = args.data_augmentation \
            if set_name == 'train' else False

        print(f"{set_name.title()} dataset contains {len(self.ids)} data "
              f"batches.")

    def __len__(self):
        return len(self.ids)
    
    def prepare_inputs(self, batch_data):
        """
        Prepare inputs to be fed to a generic model.
        """
        B, T, H, W = batch_data["input_traj_maps"].shape
        radius = min(H, W) // 45
        line_width = 2 * radius
        
        selected_inputs = {k: v.float() if \
            torch.is_tensor(v) else v for k, v in batch_data.items()}
        
        rgb_image = to_pil_image(selected_inputs["rgb_image"])
        prompted_img = torch.zeros((B, 3, H, W))
        
        lengths = torch.zeros((B, 1))
        
        for i, coord in enumerate(selected_inputs["abs_pixel_coord"][:self.args.obs_length].permute(1, 0, 2)):
            tmp_img = rgb_image.convert("RGBA")
            visual_prompt_img = Image.new(
                "RGBA", (tmp_img.width, tmp_img.height), (0, 0, 0, 0)
            )

            visual_prompt_img_canvas = ImageDraw.Draw(visual_prompt_img)
            if self.args.prompt_type == 'arrow':
                length = draw_arrow(
                    visual_prompt_img_canvas,
                    coord,
                    radius=radius,
                    line_width=line_width,
                    color=self.args.prompt_color
                )
            elif self.args.prompt_type == 'dotted':
                length = draw_dotted(
                    visual_prompt_img_canvas,
                    coord,
                    radius=radius,
                    color=self.args.prompt_color
                )
            else:
                raise NotImplementedError
            lengths[i] = length

            tmp_img = Image.alpha_composite(tmp_img, visual_prompt_img).convert(
                "RGB"
            )
            tmp_img = create_tensor_image(np.array(tmp_img), down_factor=1)
            prompted_img[i] = tmp_img
        
        selected_inputs["tensor_image"] = selected_inputs["tensor_image"].unsqueeze(0).\
            repeat(B, 1, 1, 1)
        selected_inputs["rgb_image"] = prompted_img
        
        selected_inputs["direction"] = batch_data["direction"]
        selected_inputs["lengths"] = lengths

        return selected_inputs

    def augment_traj_and_images(self, batch_data):

        image = batch_data["tensor_image"]
        rgb_image = batch_data["rgb_image"]
        abs_pixel_coord = batch_data["abs_pixel_coord"]
        input_traj_maps = batch_data["input_traj_maps"]

        # images from torch to numpy. float32 is needed by openCV
        image = image.permute(1, 2, 0).numpy().astype('float32')
        rgb_image = rgb_image.permute(1, 2, 0).numpy().astype('float32')

        # traj_maps to numpy with bs * T channels
        bs, T, old_H, old_W = input_traj_maps.shape

        input_traj_maps = input_traj_maps.view(bs * T, old_H, old_W).\
            permute(1, 2, 0).numpy().astype('float32')
            
        # keypoints to list of tuples
        # need to clamp because some slightly exit from the image
        abs_pixel_coord[:, :, 0] = np.clip(abs_pixel_coord[:, :, 0],
                                           a_min=0, a_max=old_W - 1e-3)
        abs_pixel_coord[:, :, 1] = np.clip(abs_pixel_coord[:, :, 1],
                                           a_min=0, a_max=old_H - 1e-3)
        keypoints = list(map(tuple, abs_pixel_coord.reshape(-1, 2)))

        transform = A.Compose([
            # SAFE AUGS, flips and 90rots
            A.augmentations.transforms.HorizontalFlip(p=0.5),
            A.augmentations.transforms.VerticalFlip(p=0.5),
            A.augmentations.transforms.Transpose(p=0.5),
            A.augmentations.geometric.rotate.RandomRotate90(p=1.0),

            # HIGH RISKS - HIGH PROBABILITY OF KEYPOINTS GOING OUT
            A.OneOf([  # perspective or shear
                A.augmentations.geometric.transforms.Perspective(
                    scale=0.05, pad_mode=cv2.BORDER_CONSTANT, p=1.0),
                A.augmentations.geometric.transforms.Affine(
                    shear=(-10, 10), mode=cv2.BORDER_CONSTANT, p=1.0),  # shear
            ], p=0.2),

            A.OneOf([  # translate
                A.augmentations.geometric.transforms.ShiftScaleRotate(
                    shift_limit_x=0.01, shift_limit_y=0, scale_limit=0,
                    rotate_limit=0, border_mode=cv2.BORDER_CONSTANT,
                    p=1.0),  # x translations
                A.augmentations.geometric.transforms.ShiftScaleRotate(
                    shift_limit_x=0, shift_limit_y=0.01, scale_limit=0,
                    rotate_limit=0, border_mode=cv2.BORDER_CONSTANT,
                    p=1.0),  # y translations
                A.augmentations.geometric.transforms.Affine(
                    translate_percent=(0, 0.01),
                    mode=cv2.BORDER_CONSTANT, p=1.0),  # random xy translate
            ], p=0.2),
            # random rotation
            A.augmentations.geometric.rotate.Rotate(
                limit=10, border_mode=cv2.BORDER_CONSTANT,
                p=0.4),
        ],
            keypoint_params=A.KeypointParams(format='xy',
                                             remove_invisible=False),
            additional_targets={'traj_map': 'image', 'rgb_image': 'image'},
        )
        transformed = transform(
            image=image, keypoints=keypoints, traj_map=input_traj_maps, rgb_image=rgb_image)

        # FROM NUMPY BACK TO TENSOR
        image = torch.tensor(transformed['image']).permute(2, 0, 1)
        rgb_image = torch.tensor(transformed['rgb_image']).permute(2, 0, 1)
        C, new_H, new_W = image.shape
        abs_pixel_coord = torch.tensor(transformed['keypoints']).\
            view(batch_data["abs_pixel_coord"].shape)
        input_traj_maps = torch.tensor(transformed['traj_map']).\
            permute(2, 0, 1).view(bs, T, new_H, new_W)

        # NEW AUGMENTATION: INVERT TIME
        if random.random() > 0.5:
            abs_pixel_coord = abs_pixel_coord.flip(dims=(0,))
            input_traj_maps = input_traj_maps.flip(dims=(1,))
            
        batch_data["tensor_image"] = image
        batch_data["rgb_image"] = rgb_image
        batch_data["abs_pixel_coord"] = abs_pixel_coord
        batch_data["input_traj_maps"] = input_traj_maps
        return batch_data
    
    def __getitem__(self, i):
        batch_path = os.path.join(self.path_to_folder, self.ids[i])

        with open(batch_path, 'rb') as f:
            batch_data, batch_id = pickle.load(f)

        # rescale coordinates wrt pixel coordinates
        batch_data["abs_pixel_coord"] /= self.args.down_factor
        batch_data = {k: torch.tensor(v) if not torch.is_tensor(v) else v for k, v in batch_data.items()}

        if self.data_augmentation:
            batch_data = self.augment_traj_and_images(batch_data)
        batch_data = self.prepare_inputs(batch_data)
        
        ground_truth = batch_data["abs_pixel_coord"]

        return batch_data, ground_truth, batch_id


# -

def get_dataloader(args, set_name):
    """
    Create a data loader for a specific set/data split
    """
    assert set_name in ['train', 'valid', 'test']

    shuffle = args.shuffle_train_batches if set_name == 'train' else \
        args.shuffle_test_batches

    dataset = dataset_set_name(args, set_name=set_name)
    loader = DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=2)

    return loader
