import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import DataLoader
from utils.homography import generate_homography, world2image
from PIL import Image, ImageDraw
import cv2
import torchvision.transforms as TT
import random
import albumentations as A
from tqdm import tqdm


IMAGE_SCALE_DOWN = 0.25


def get_dataloader(data_dir, phase, obs_len, pred_len, batch_size, args, type='traj'):
    r"""Get dataloader for a specific phase

    Args:
        data_dir (str): path to the dataset directory
        phase (str): phase of the data, one of 'train', 'val', 'test'
        obs_len (int): length of observed trajectory
        pred_len (int): length of predicted trajectory
        batch_size (int): batch size

    Returns:
        loader_phase (torch.utils.data.DataLoader): dataloader for the specific phase
    """

    assert phase in ['train', 'val', 'test']

    data_set = data_dir + '/' + phase + '/'
    shuffle = True if phase == 'train' else False
    drop_last = True if phase == 'train' else False

    if type == 'traj':
        pass
    elif type == 'goal':
        # dataset_phase = GoalDataset(data_set, obs_len=obs_len, pred_len=pred_len, batch_size=batch_size, phase=phase)
        dataset_phase = GoalDataset(data_set, obs_len=obs_len, pred_len=pred_len, batch_size=1, phase=phase, args=args)
        loader_phase = DataLoader(dataset_phase, batch_size=1, shuffle=shuffle)
    return loader_phase


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non-linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def load_semantic_map(path):
        sem_map = cv2.imread(path, flags=0)
        # from (X,Y) valued in [0,C] to (X,Y,C) valued in [0,1]
        num_classes = 6
        sem_map = [(sem_map == v) for v in range(num_classes)]
        sem_map = np.stack(sem_map, axis=-1).astype(int)
        return sem_map

def create_tensor_image(big_numpy_image,
                        down_factor=1):
    img = TT.functional.to_tensor(big_numpy_image)
    C, H, W = img.shape
    new_heigth = int(H * down_factor)
    new_width = int(W * down_factor)
    tensor_image = TT.functional.resize(img, (new_heigth, new_width),
                                        interpolation=TT.InterpolationMode.NEAREST)
    return tensor_image



def normalize_prob_map(x):
    """Normalize a probability map of shape (B, T, H, W) so
    that sum over H and W equal ones"""
    assert len(x.shape) == 4
    sums = x.sum(-1, keepdim=True).sum(-2, keepdim=True)
    x = torch.divide(x, sums)
    return x


def un_normalize_prob_map(x):
    """Un-Normalize a probability map of shape (B, T, H, W) so
    that each pixel has value between 0 and 1"""
    assert len(x.shape) == 4
    (B, T, H, W) = x.shape
    maxs, _ = x.reshape(B, T, -1).max(-1)
    x = torch.divide(x, maxs.unsqueeze(-1).unsqueeze(-1))
    return x

def make_gaussian_map_patches(gaussian_centers,
                              width,
                              height,
                              norm=False,
                              gaussian_std=None):
    """
    gaussian_centers.shape == (T, 2)
    Make a PyTorch gaussian GT map of size (1, T, height, width)
    centered in gaussian_centers. The coordinates of the centers are
    computed starting from the left upper corner.
    """
    assert isinstance(gaussian_centers, torch.Tensor)

    if not gaussian_std:
        gaussian_std = min(width, height) / 64
    gaussian_var = gaussian_std ** 2

    x_range = torch.arange(0, height, 1)
    y_range = torch.arange(0, width, 1)
    grid_x, grid_y = torch.meshgrid(x_range, y_range)
    pos = torch.stack((grid_y, grid_x), dim=2)
    pos = pos.unsqueeze(2)

    gaussian_map = (1. / (2. * math.pi * gaussian_var)) * \
                   torch.exp(-torch.sum((pos - gaussian_centers) ** 2., dim=-1)
                             / (2 * gaussian_var))

    # from (H, W, T) to (1, T, H, W)
    gaussian_map = gaussian_map.permute(2, 0, 1).unsqueeze(0)

    if norm:
        # normalised prob: sum over coordinates equals 1
        gaussian_map = normalize_prob_map(gaussian_map)
    else:
        # un-normalize probabilities (otherwise the network learns all zeros)
        # each pixel has value between 0 and 1
        gaussian_map = un_normalize_prob_map(gaussian_map)

    return gaussian_map


def create_CNN_inputs_loop(batch_abs_pixel_coords,
                           tensor_image):

    num_agents = batch_abs_pixel_coords.shape[0]
    C, H, W = tensor_image.shape
    input_traj_maps = list()

    # loop over agents
    for agent_idx in range(num_agents):
        trajectory = batch_abs_pixel_coords[agent_idx].detach()\
            .clone().to(torch.device("cpu"))

        traj_map_cnn = make_gaussian_map_patches(
            gaussian_centers=trajectory,
            height=H,
            width=W)
        # append
        input_traj_maps.append(traj_map_cnn)


    # list --> tensor
    input_traj_maps = torch.cat(input_traj_maps, dim=0)

    return input_traj_maps

    

class GoalDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.02, min_ped=1, delim='\t', batch_size=8, phase='test', args=None):
        """
        Args:
        - data_dir: Directory containing dataset files in the format <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non-linear traj when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a sequence
        - delim: Delimiter in the dataset files
        """
        super(GoalDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.batch_size = batch_size
        self.data_augmentation = phase == 'train'
        self.args = args

        all_files = sorted(os.listdir(self.data_dir))
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]

        self.homography = {}
        self.scene_img_dict = {}
        self.scene_sem_dict = {}
        scene_img_map = {'biwi_eth': 'seq_eth', 'biwi_hotel': 'seq_hotel',
                         'students001': 'students003', 'students003': 'students003', 'uni_examples': 'students003',
                         'crowds_zara01': 'crowds_zara01', 'crowds_zara02': 'crowds_zara02', 'crowds_zara03': 'crowds_zara02'}
        
        self.seq_coord = []
        self.scene_id = []
        self.scene_img = []
        self.scene_sem = []
        self.input_traj_maps = []

        for path in all_files:
            print(path)
            seq_list = []
            input_maps = []
            # scene_img = []
            # Load image
            parent_dir, scene_name = os.path.split(path)
            parent_dir, phase = os.path.split(parent_dir)
            parent_dir, dataset_name = os.path.split(parent_dir)
            scene_name, _ = os.path.splitext(scene_name)
            scene_name = scene_name.replace('_' + phase, '')

            try:
                # self.scene_img[scene_name] = Image.open(os.path.join(parent_dir, "image", scene_img_map[scene_name] + "_reference.png"))
                self.scene_img_dict[scene_name] = create_tensor_image(Image.open(os.path.join(parent_dir, "image", scene_img_map[scene_name] + "_bg.png")).convert("RGB"), down_factor=IMAGE_SCALE_DOWN)
                self.scene_sem_dict[scene_name] = create_tensor_image(load_semantic_map(os.path.join(parent_dir, "image", scene_img_map[scene_name] + "_oracle.png")), down_factor=IMAGE_SCALE_DOWN)
             
            except:
                self.scene_img_dict[scene_name] = None
                self.scene_sem_dict[scene_name] = None
            
            # Load homography matrix
            if dataset_name in ["eth", "hotel", "univ", "zara1", "zara2", "rawall"]:
                homography_file = os.path.join(parent_dir, "homography", scene_name + "_H.txt")
                self.homography[scene_name] = np.loadtxt(homography_file)
                
            for k, v in self.homography.items():
                self.homography[k] = v.copy() @ generate_homography(scale=IMAGE_SCALE_DOWN)

            # Load data
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            H = torch.tensor(self.homography[scene_name]).type(torch.float)
            
            for idx in tqdm(range(0, num_sequences * self.skip + 1, skip)):
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                num_peds_considered = 0
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    seq = world2image(torch.from_numpy(curr_seq[:num_peds_considered]).type(torch.float).permute(0, 2, 1), H)
                    seq_list.append(seq)
                    
                    input_map = create_CNN_inputs_loop(batch_abs_pixel_coords=seq, tensor_image=self.scene_sem_dict[scene_name])
                    input_maps.append(input_map)
                    
                    
                    # for seq in curr_seq[:num_peds_considered]:
                    #     tmp_seq = torch.from_numpy(seq).type(torch.float).permute(1, 0)[:obs_len]
                    #     tmp_seq = world2image(tmp_seq, H)
                    #     tmp_img = img.convert("RGBA")
                    #     visual_prompt_img = Image.new("RGBA", (tmp_img.width, tmp_img.height), (0, 0, 0, 0))
                        
                    #     visual_prompt_img_canvas = ImageDraw.Draw(visual_prompt_img)
                    #     draw_arrow(visual_prompt_img_canvas, tmp_seq)
                        
                    #     tmp_img = Image.alpha_composite(tmp_img, visual_prompt_img).convert("RGB")
                    #     tmp_img = create_tensor_image(np.array(tmp_img), down_factor=1)
                    #     scene_img.append(tmp_img)
                        
                    
            # seq_list = np.concatenate(seq_list, axis=0)
            
            # seq_coord = torch.from_numpy(seq_list).type(torch.float).permute(0, 2, 1)  # NTC
            
            # seq_coord = world2image(seq_coord, H)
            
            # seq_coord = torch.split(seq_coord, self.batch_size, dim=0)
            self.seq_coord.extend(seq_list)
            self.input_traj_maps.extend(input_maps)
            
            num_batch = len(seq_list)
            scene_id = [scene_name] * num_batch
            self.scene_id.extend(scene_id)
            
            # scene_img = torch.stack(scene_img)
            # scene_img = torch.split(scene_img, self.batch_size, dim=0)
            scene_img = [self.scene_img_dict[scene_name] for _ in range(num_batch)]
            self.scene_img.extend(scene_img)
            
            scene_sem = [self.scene_sem_dict[scene_name] for _ in range(num_batch)]
            self.scene_sem.extend(scene_sem)
            
    def augment_traj_and_images(self, batch_data):
        image = batch_data["scene_sem"]
        rgb_image = batch_data["scene_img"]
        abs_pixel_coord = batch_data["seq_coord"]
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
            view(batch_data["seq_coord"].shape)
        input_traj_maps = torch.tensor(transformed['traj_map']).\
            permute(2, 0, 1).view(bs, T, new_H, new_W)

        # NEW AUGMENTATION: INVERT TIME
        if random.random() > 0.5:
            abs_pixel_coord = abs_pixel_coord.flip(dims=(0,))
            input_traj_maps = input_traj_maps.flip(dims=(1,))
        
        batch_data["scene_sem"] = image
        batch_data["scene_img"] = rgb_image
        batch_data["seq_coord"] = abs_pixel_coord
        batch_data["input_traj_maps"] = input_traj_maps

        return batch_data


    def __len__(self):
        return len(self.seq_coord)

    def __getitem__(self, index):
        out = {"seq_coord": self.seq_coord[index],
               'scene_img': self.scene_img[index],
               'scene_sem': self.scene_sem[index],
               "scene_id": self.scene_id[index],
               "input_traj_maps": self.input_traj_maps[index]}
        if self.data_augmentation:
            out = self.augment_traj_and_images(out)
        if self.args.minus_neighbor:
            sum_traj_maps = out["input_traj_maps"][:,:8].sum(dim=0)
            for i in range(out["input_traj_maps"].shape[0]):
                out["input_traj_maps"][i,:8] = 2 * out["input_traj_maps"][i,:8] - sum_traj_maps
        return out
