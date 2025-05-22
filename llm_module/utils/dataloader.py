import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import DataLoader
from .homography import generate_homography, world2image
from PIL import Image, ImageDraw
import cv2
import torchvision.transforms as TT
import random
import albumentations as A
from tqdm import tqdm


def get_dataloader(
    data_dir,
    phase,
    obs_len,
    pred_len,
    batch_size,
    down_factor=4,
    type="traj",
    augment=True,
    goal_aug_prob=-1,
):
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

    assert phase in ["train", "val", "test"]

    data_set = data_dir + "/" + phase + "/"
    shuffle = True if phase == "train" and augment else False
    drop_last = True if phase == "train" and augment else False

    if type == "traj":
        dataset_phase = TrajectoryDataset(
            data_set, obs_len=obs_len, pred_len=pred_len, down_factor=down_factor
        )
        sampler_phase = None
        if batch_size > 1:
            sampler_phase = TrajBatchSampler(
                dataset_phase,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        loader_phase = DataLoader(
            dataset_phase,
            collate_fn=traj_collate_fn,
            batch_sampler=sampler_phase,
            pin_memory=True,
        )
    elif type == "goal":
        dataset_phase = GoalDataset(
            data_set,
            obs_len=obs_len,
            pred_len=pred_len,
            batch_size=batch_size,
            phase=phase,
            down_factor=down_factor,
            augment=augment,
            goal_aug_prob=goal_aug_prob,
        )
        loader_phase = DataLoader(dataset_phase, batch_size=1, shuffle=shuffle)
    return loader_phase


def traj_collate_fn(data):
    r"""Collate function for the dataloader

    Args:
        data (list): list of tuples of (obs_seq, pred_seq, non_linear_ped, loss_mask, seq_start_end, scene_id)

    Returns:
        obs_seq_list (torch.Tensor): (num_ped, obs_len, 2)
        pred_seq_list (torch.Tensor): (num_ped, pred_len, 2)
        non_linear_ped_list (torch.Tensor): (num_ped,)
        loss_mask_list (torch.Tensor): (num_ped, obs_len + pred_len)
        scene_mask (torch.Tensor): (num_ped, num_ped)
        seq_start_end (torch.Tensor): (num_ped, 2)
        scene_id
    """

    data_collated = {}
    for k in data[0].keys():
        data_collated[k] = [d[k] for d in data]

    _len = [len(seq) for seq in data_collated["obs_traj"]]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]
    seq_start_end = torch.LongTensor(seq_start_end)
    scene_mask = torch.zeros(sum(_len), sum(_len), dtype=torch.bool)
    for idx, (start, end) in enumerate(seq_start_end):
        scene_mask[start:end, start:end] = 1

    data_collated["obs_traj"] = torch.cat(data_collated["obs_traj"], dim=0)
    data_collated["pred_traj"] = torch.cat(data_collated["pred_traj"], dim=0)
    data_collated["non_linear_ped"] = torch.cat(data_collated["non_linear_ped"], dim=0)
    data_collated["loss_mask"] = torch.cat(data_collated["loss_mask"], dim=0)
    data_collated["scene_mask"] = scene_mask
    data_collated["seq_start_end"] = seq_start_end
    data_collated["frame"] = torch.cat(data_collated["frame"], dim=0)
    data_collated["scene_id"] = np.concatenate(data_collated["scene_id"], axis=0)

    return data_collated


def goal_collate_fn(data):
    r"""Collate function for the dataloader

    Args:
        data (list): list of tuples of (obs_seq, pred_seq, non_linear_ped, loss_mask, seq_start_end, scene_id)

    Returns:
        obs_seq_list (torch.Tensor): (num_ped, obs_len, 2)
        pred_seq_list (torch.Tensor): (num_ped, pred_len, 2)
        non_linear_ped_list (torch.Tensor): (num_ped,)
        loss_mask_list (torch.Tensor): (num_ped, obs_len + pred_len)
        scene_mask (torch.Tensor): (num_ped, num_ped)
        seq_start_end (torch.Tensor): (num_ped, 2)
        scene_id
    """

    data_collated = {}

    data_collated["obs_traj"] = torch.cat(data_collated["obs_traj"], dim=0)
    data_collated["goal_traj"] = torch.cat(data_collated["goal_traj"], dim=0)
    data_collated["scene_img"] = torch.cat(data_collated["scene_img"], dim=0)
    data_collated["scene_sem"] = torch.cat(data_collated["scene_sem"], dim=0)
    data_collated["scene_id"] = np.concatenate(data_collated["scene_id"], axis=0)

    return data_collated


class TrajBatchSampler(Sampler):
    r"""Samples batched elements by yielding a mini-batch of indices.
    Args:
        data_source (Dataset): dataset to sample from
        batch_size (int): Size of mini-batch.
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
        generator (Generator): Generator used in sampling.
    """

    def __init__(
        self, data_source, batch_size=64, shuffle=False, drop_last=False, generator=None
    ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

    def __iter__(self):
        assert len(self.data_source) == len(self.data_source.num_peds_in_seq)

        if self.shuffle:
            if self.generator is None:
                generator = torch.Generator()
                generator.manual_seed(
                    int(torch.empty((), dtype=torch.int64).random_().item())
                )
            else:
                generator = self.generator
            indices = torch.randperm(
                len(self.data_source), generator=generator
            ).tolist()
        else:
            indices = list(range(len(self.data_source)))
        num_peds_indices = self.data_source.num_peds_in_seq[indices]

        batch = []
        total_num_peds = 0
        for idx, num_peds in zip(indices, num_peds_indices):
            batch.append(idx)
            total_num_peds += num_peds
            if total_num_peds >= self.batch_size:
                yield batch
                batch = []
                total_num_peds = 0
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        # Approximated number of batches.
        # The order of trajectories can be shuffled, so this number can vary from run to run.
        if self.drop_last:
            return sum(self.data_source.num_peds_in_seq) // self.batch_size
        else:
            return (
                sum(self.data_source.num_peds_in_seq) + self.batch_size - 1
            ) // self.batch_size


def read_file(_path, delim="\t", dataset_name='eth'):
    data = []
    if delim == "tab":
        delim = "\t"
    elif delim == "space":
        delim = " "
    if dataset_name == 'sdd':
        delim = " "

    with open(_path, "r") as f:
        for line in f:
            line = line.strip().split(delim)
            if dataset_name == 'sdd':
                if line[-1] == 'Pedestrian':
                    line = line[:-1]
                else:
                    continue
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
    num_classes = 2
    sem_map = [(sem_map == v) for v in range(num_classes)]
    sem_map = np.stack(sem_map, axis=-1).astype(int)
    return sem_map


def create_tensor_image(big_numpy_image, down_factor=1):
    img = TT.functional.to_tensor(big_numpy_image)
    C, H, W = img.shape
    new_heigth = int(H * down_factor)
    new_width = int(W * down_factor)
    tensor_image = TT.functional.resize(
        img, (new_heigth, new_width), interpolation=TT.InterpolationMode.NEAREST
    )
    return tensor_image


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
        self,
        data_dir,
        obs_len=8,
        pred_len=12,
        down_factor=4,
        skip=1,
        threshold=0.02,
        min_ped=1,
        delim="\t",
    ):
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
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = sorted(os.listdir(self.data_dir))
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        loss_mask_list = []
        non_linear_ped = []
        frame_list = []
        scene_id = []
        self.homography = {}
        self.scene_img = {}
        self.scene_desc = {}
        scene_img_map = {
            "biwi_eth": "seq_eth",
            "biwi_hotel": "seq_hotel",
            "students001": "students003",
            "students003": "students003",
            "uni_examples": "students003",
            "crowds_zara01": "crowds_zara01",
            "crowds_zara02": "crowds_zara02",
            "crowds_zara03": "crowds_zara02",
            "bookstore_0": "bookstore_0",
            "bookstore_1": "bookstore_1",
            "bookstore_2": "bookstore_2",
            "bookstore_3": "bookstore_3",
            "coupa_0": "coupa_0",
            "coupa_1": "coupa_1",
            "coupa_3": "coupa_3",
            "deathCircle_0": "deathCircle_0",
            "deathCircle_1": "deathCircle_1",
            "deathCircle_2": "deathCircle_2",
            "deathCircle_3": "deathCircle_3",
            "deathCircle_4": "deathCircle_4",
            "gates_0": "gates_0",
            "gates_1": "gates_1",
            "gates_2": "gates_2",
            "gates_3": "gates_3",
            "gates_4": "gates_4",
            "gates_5": "gates_5",
            "gates_6": "gates_6",
            "gates_7": "gates_7",
            "gates_8": "gates_8",
            "hyang_0": "hyang_0",
            "hyang_1": "hyang_1",
            "hyang_3": "hyang_3",
            "hyang_4": "hyang_4",
            "hyang_5": "hyang_5",
            "hyang_6": "hyang_6",
            "hyang_7": "hyang_7",
            "hyang_8": "hyang_8",
            "hyang_9": "hyang_9",
            "nexus_0": "nexus_0",
            "nexus_1": "nexus_1",
            "nexus_2": "nexus_2",
            "nexus_3": "nexus_3",
            "nexus_4": "nexus_4",
            "nexus_5": "nexus_5",
            "nexus_6": "nexus_6",
            "nexus_7": "nexus_7",
            "nexus_8": "nexus_8",
            "nexus_9": "nexus_9",
            "nexus_10": "nexus_10",
            "nexus_11": "nexus_10",
            "quad_0": "quad_0",
            "quad_1": "quad_1",
            "quad_2": "quad_2",
            "quad_3": "quad_3",
            
        }

        for path in all_files:
            # Load image
            parent_dir, scene_name = os.path.split(path)
            parent_dir, phase = os.path.split(parent_dir)
            parent_dir, dataset_name = os.path.split(parent_dir)
            scene_name, _ = os.path.splitext(scene_name)
            scene_name = scene_name.replace("_" + phase, "")
        
            try:
                # self.scene_img[scene_name] = Image.open(
                #     os.path.join(
                #         parent_dir,
                #         "image",
                #         scene_img_map[scene_name] + "_reference.png",
                #     )
                # )
                scene_img = Image.open(
                    os.path.join(
                        parent_dir,
                        "image",
                        scene_img_map[scene_name] + "_bg.png",
                    )
                )
                width, height = scene_img.size
                self.scene_img[scene_name] = scene_img.resize(
                    (int(width / down_factor), int(height / down_factor))
                )

                # check caption file exist
                if os.path.exists(
                    os.path.join(
                        parent_dir, "image", scene_img_map[scene_name] + "_caption.txt"
                    )
                ):
                    with open(
                        os.path.join(
                            parent_dir,
                            "image",
                            scene_img_map[scene_name] + "_caption.txt",
                        ),
                        "r",
                    ) as f:
                        self.scene_desc[scene_name] = f.read()
                else:
                    self.scene_desc[scene_name] = ""
            except:
                print(scene_name)
                self.scene_img[scene_name] = None
                self.scene_desc[scene_name] = ""

            # Load homography matrix
            if dataset_name in ["eth", "hotel", "univ", "zara1", "zara2", "rawall"]:
                homography_file = os.path.join(
                    parent_dir, "homography", scene_name + "_H.txt"
                )
                self.homography[scene_name] = np.loadtxt(homography_file)
            else:
                self.homography[scene_name] = np.eye(3)

            # Load data
            data = read_file(path, delim, dataset_name=dataset_name)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))
            

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx : idx + self.seq_len], axis=0
                )
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len or curr_ped_seq.shape[0] != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    frame_list.extend([frames[idx]] * num_peds_considered)
                    scene_id.extend([scene_name] * num_peds_considered)

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        self.num_peds_in_seq = np.array(num_peds_in_seq)
        self.frame_list = np.array(frame_list, dtype=np.int32)
        self.scene_id = np.array(scene_id)

        # Convert numpy -> Torch Tensor
        self.obs_traj = (
            torch.from_numpy(seq_list[:, :, : self.obs_len])
            .type(torch.float)
            .permute(0, 2, 1)
        )  # NTC
        self.pred_traj = (
            torch.from_numpy(seq_list[:, :, self.obs_len :])
            .type(torch.float)
            .permute(0, 2, 1)
        )  # NTC
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float).gt(0.5)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float).gt(0.5)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        self.frame_list = torch.from_numpy(self.frame_list).type(torch.long)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = {
            "obs_traj": self.obs_traj[start:end],
            "pred_traj": self.pred_traj[start:end],
            "non_linear_ped": self.non_linear_ped[start:end],
            "loss_mask": self.loss_mask[start:end],
            "scene_mask": None,
            "seq_start_end": [[0, end - start]],
            "frame": self.frame_list[start:end],
            "scene_id": self.scene_id[start:end],
        }
        return out


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


def make_gaussian_map_patches(
    gaussian_centers, width, height, norm=False, gaussian_std=None
):
    """
    gaussian_centers.shape == (T, 2)
    Make a PyTorch gaussian GT map of size (1, T, height, width)
    centered in gaussian_centers. The coordinates of the centers are
    computed starting from the left upper corner.
    """
    assert isinstance(gaussian_centers, torch.Tensor)

    if not gaussian_std:
        gaussian_std = min(width, height) / 64
    else:
        gaussian_std = min(width, height) * gaussian_std
    gaussian_var = gaussian_std**2

    x_range = torch.arange(0, height, 1)
    y_range = torch.arange(0, width, 1)
    grid_x, grid_y = torch.meshgrid(x_range, y_range)
    pos = torch.stack((grid_y, grid_x), dim=2)
    pos = pos.unsqueeze(2)

    gaussian_map = (1.0 / (2.0 * math.pi * gaussian_var)) * torch.exp(
        -torch.sum((pos - gaussian_centers) ** 2.0, dim=-1) / (2 * gaussian_var)
    )

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


def make_arc_gaussian_map_patches(
    circle_center, gaussian_center, width, height, arc_angle, non_walkable_area, gaussian_std=None
):
    if circle_center[0] == gaussian_center[0] and circle_center[1] == gaussian_center[1]:
        return make_gaussian_map_patches(
            gaussian_centers=gaussian_center, width=width, height=height, gaussian_std=gaussian_std
        )
    if not gaussian_std:
        gaussian_std = min(width, height) / 64
    else:
        gaussian_std = min(width, height) * gaussian_std
    
    gaussian_var = gaussian_std**2

    vec = gaussian_center - circle_center
    vec_norm = vec / torch.norm(vec)

    perp_vec = torch.tensor([-vec_norm[1], vec_norm[0]])

    angle_rad = math.radians(arc_angle)

    arc_points = []
    for theta in torch.linspace(-angle_rad/2, angle_rad/2, steps=20):
        rotation_matrix = torch.tensor([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)]
        ])
        arc_point = circle_center + rotation_matrix @ vec_norm * torch.norm(vec)
        arc_points.append(arc_point)

    arc_points = torch.stack(arc_points)

    x_range = torch.arange(0, height, 1)
    y_range = torch.arange(0, width, 1)
    grid_x, grid_y = torch.meshgrid(x_range, y_range)
    pos = torch.stack((grid_y, grid_x), dim=2)

    gaussian_map = torch.zeros(height, width)
    for arc_point in arc_points:
        pos_diff = pos - arc_point.unsqueeze(0).unsqueeze(0)
        single_gaussian = (1.0 / (2.0 * math.pi * gaussian_var)) * torch.exp(
            -torch.sum(pos_diff ** 2.0, dim=-1) / (2 * gaussian_var)
        )
        gaussian_map += single_gaussian
    
    gaussian_map /= torch.max(gaussian_map)
    gaussian_map[non_walkable_area == 1] = 0

    return gaussian_map.unsqueeze(0)


def create_CNN_inputs_loop(batch_abs_pixel_coords, tensor_image, std=None):

    num_agents = batch_abs_pixel_coords.shape[0]
    C, H, W = tensor_image.shape
    input_traj_maps = list()

    # loop over agents
    for agent_idx in range(num_agents):
        trajectory = (
            batch_abs_pixel_coords[agent_idx].detach().clone().to(torch.device("cpu"))
        )

        traj_map_cnn = make_gaussian_map_patches(
            gaussian_centers=trajectory, height=H, width=W, gaussian_std=std
        )
        # append
        input_traj_maps.append(traj_map_cnn)

    # list --> tensor
    input_traj_maps = torch.cat(input_traj_maps, dim=0)

    return input_traj_maps


RADIUS = 2
LINE_WIDTH = 3


def draw_arrow(draw, points, color="red"):

    total_lenght = 0
    for point1, point2 in zip(points[:-2], points[1:-1]):
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]
        draw.ellipse([x1 - RADIUS, y1 - RADIUS, x1 + RADIUS, y1 + RADIUS], fill=color)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=LINE_WIDTH)
        if not (x1 == x2 and y1 == y2):
            total_lenght += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    x1, y1 = points[-2, 0], points[-2, 1]
    x2, y2 = points[-1, 0], points[-1, 1]
    dx = x2 - x1
    dy = y2 - y1
    angle = math.atan2(dy, dx)

    total_lenght += math.sqrt((dx) ** 2 + (dy) ** 2)
    arrowhead_length = LINE_WIDTH * 3
    if total_lenght < arrowhead_length:
        draw.line([(x1, y1), (x2, y2)], fill=color, width=LINE_WIDTH)
        return

    end_x = x2 - arrowhead_length * 0.8 * math.cos(angle)
    end_y = y2 - arrowhead_length * 0.8 * math.sin(angle)

    draw.line([(x1, y1), (end_x, end_y)], fill=color, width=LINE_WIDTH)

    left_angle = angle + math.pi / 6
    right_angle = angle - math.pi / 6

    left_arrowhead = (
        x2 - arrowhead_length * math.cos(left_angle),
        y2 - arrowhead_length * math.sin(left_angle),
    )
    right_arrowhead = (
        x2 - arrowhead_length * math.cos(right_angle),
        y2 - arrowhead_length * math.sin(right_angle),
    )

    draw.polygon([(x2, y2), left_arrowhead, right_arrowhead], fill=color)


class GoalDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
        self,
        data_dir,
        obs_len=8,
        pred_len=12,
        down_factor=4,
        skip=1,
        threshold=0.02,
        min_ped=1,
        delim="\t",
        batch_size=8,
        phase="test",
        augment=True,
        goal_aug_prob=-1,
    ):
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
        self.data_augmentation = phase == "train" and augment
        self.goal_aug_prob = goal_aug_prob

        all_files = sorted(os.listdir(self.data_dir))
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]

        self.homography = {}
        self.scene_img_dict = {}
        self.scene_sem_dict = {}
        scene_img_map = {
            "biwi_eth": "seq_eth",
            "biwi_hotel": "seq_hotel",
            "students001": "students003",
            "students003": "students003",
            "uni_examples": "students003",
            "crowds_zara01": "crowds_zara01",
            "crowds_zara02": "crowds_zara02",
            "crowds_zara03": "crowds_zara02",
            "bookstore_0": "bookstore_0",
            "bookstore_1": "bookstore_1",
            "bookstore_2": "bookstore_2",
            "bookstore_3": "bookstore_3",
            "coupa_0": "coupa_0",
            "coupa_1": "coupa_1",
            "coupa_3": "coupa_3",
            "deathCircle_0": "deathCircle_0",
            "deathCircle_1": "deathCircle_1",
            "deathCircle_2": "deathCircle_2",
            "deathCircle_3": "deathCircle_3",
            "deathCircle_4": "deathCircle_4",
            "gates_0": "gates_0",
            "gates_1": "gates_1",
            "gates_2": "gates_2",
            "gates_3": "gates_3",
            "gates_4": "gates_4",
            "gates_5": "gates_5",
            "gates_6": "gates_6",
            "gates_7": "gates_7",
            "gates_8": "gates_8",
            "hyang_0": "hyang_0",
            "hyang_1": "hyang_1",
            "hyang_3": "hyang_3",
            "hyang_4": "hyang_4",
            "hyang_5": "hyang_5",
            "hyang_6": "hyang_6",
            "hyang_7": "hyang_7",
            "hyang_8": "hyang_8",
            "hyang_9": "hyang_9",
        }

        self.seq_coord = []
        self.scene_id = []
        self.scene_img = []
        self.scene_sem = []
        self.input_traj_maps = []
        self.label_maps = []

        for path in all_files:
            print(path)
            seq_list = []
            # scene_img = []
            # Load image
            parent_dir, scene_name = os.path.split(path)
            parent_dir, phase = os.path.split(parent_dir)
            parent_dir, dataset_name = os.path.split(parent_dir)
            scene_name, _ = os.path.splitext(scene_name)
            scene_name = scene_name.replace("_" + phase, "")

            try:
                # self.scene_img[scene_name] = Image.open(os.path.join(parent_dir, "image", scene_img_map[scene_name] + "_reference.png"))
                self.scene_img_dict[scene_name] = create_tensor_image(
                    Image.open(
                        os.path.join(
                            parent_dir, "image", scene_img_map[scene_name] + "_bg.png"
                        )
                    ).convert("RGB"),
                    down_factor=1 / down_factor,
                )
                self.scene_sem_dict[scene_name] = create_tensor_image(
                    load_semantic_map(
                        os.path.join(
                            parent_dir,
                            "image",
                            scene_img_map[scene_name] + "_oracle.png",
                        )
                    ),
                    down_factor=1 / down_factor,
                )
                # self.scene_sem_dict[scene_name] = create_tensor_image(
                #     load_semantic_map(
                #         os.path.join(
                #             parent_dir, "image", scene_img_map[scene_name] + "_sem.png"
                #         )
                #     ),
                #     down_factor=1 / down_factor,
                # )

            except:
                self.scene_img_dict[scene_name] = None
                self.scene_sem_dict[scene_name] = None

            # Load homography matrix
            if dataset_name in ["eth", "hotel", "univ", "zara1", "zara2", "rawall"]:
                homography_file = os.path.join(
                    parent_dir, "homography", scene_name + "_H.txt"
                )
                self.homography[scene_name] = np.loadtxt(
                    homography_file
                ) @ generate_homography(scale=1 / down_factor)

            # Load data
            data = read_file(path, delim, dataset_name=dataset_name)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            if dataset_name == 'sdd':
                H = torch.tensor(generate_homography(scale=1 / down_factor)).type(torch.float)
            else:
                H = torch.tensor(self.homography[scene_name]).type(torch.float)

            for idx in tqdm(range(0, num_sequences * self.skip + 1, skip)):
                curr_seq_data = np.concatenate(
                    frame_data[idx : idx + self.seq_len], axis=0
                )
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                num_peds_considered = 0
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len or curr_ped_seq.shape[0] != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    seq_list.append(curr_seq[:num_peds_considered])

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

            seq_list = np.concatenate(seq_list, axis=0)

            seq_coord = (
                torch.from_numpy(seq_list).type(torch.float).permute(0, 2, 1)
            )  # NTC

            
            seq_coord = world2image(seq_coord, H)
            input_traj_maps = create_CNN_inputs_loop(
                batch_abs_pixel_coords=seq_coord,
                tensor_image=self.scene_sem_dict[scene_name],
            )
        
            seq_coord = torch.split(seq_coord, self.batch_size, dim=0)
            self.seq_coord.extend(seq_coord)
            self.input_traj_maps.extend(
                torch.split(input_traj_maps, self.batch_size, dim=0)
            )

            num_batch = len(seq_coord)
            scene_id = [scene_name] * num_batch
            self.scene_id.extend(scene_id)

            # scene_img = torch.stack(scene_img)
            # scene_img = torch.split(scene_img, self.batch_size, dim=0)
            scene_img = [self.scene_img_dict[scene_name] for _ in range(num_batch)]
            self.scene_img.extend(scene_img)

            scene_sem = [self.scene_sem_dict[scene_name] for _ in range(num_batch)]
            self.scene_sem.extend(scene_sem)

    def augment_goal(self, last_positions, goal_positions, semantic_map, max_angle=45):
        with torch.no_grad():
            B = last_positions.shape[0]
            _, H, W = semantic_map.shape
            max_distance_aug = min(H, W) * 0.1
            
            relative_positions = goal_positions - last_positions  # (B, 2)

            # Convert to polar coordinates
            distances = (torch.square(relative_positions) + 1e-6).sum(dim=1).sqrt()  # (B,)
            angles = torch.atan2(relative_positions[:, 1], relative_positions[:, 0])  # (B,)
            
            random_angles = (torch.rand(B) * 2 - 1) * max_angle * (np.pi / 180)  # (B,)
            new_angles = angles + random_angles
            
            random_distances = (torch.rand(B) * 2 - 1) * max_distance_aug  # (B,)
            new_distances = distances + random_distances
            
            new_relative_positions = torch.stack(
                [new_distances * torch.cos(new_angles), new_distances * torch.sin(new_angles)], dim=1
            ) # (B, 2)
            
            new_goal_positions = last_positions + new_relative_positions
            
            new_goal_positions = new_goal_positions.round()
            
            for i in range(B):
                new_pos = new_goal_positions[i].long()
                if not (0 <= new_pos[1] < H and 0 <= new_pos[0] < W) or (semantic_map[0, new_pos[1], new_pos[0]] == 1):
                    new_goal_positions[i] = goal_positions[i]
        
        new_goal_positions.requires_grad = True
        return new_goal_positions
        

    def augment_traj_and_images(self, batch_data):
        image = batch_data["scene_sem"]
        rgb_image = batch_data["scene_img"]
        abs_pixel_coord = batch_data["seq_coord"]
        input_traj_maps = batch_data["input_traj_maps"]

        # images from torch to numpy. float32 is needed by openCV
        image = image.permute(1, 2, 0).numpy().astype("float32")
        rgb_image = rgb_image.permute(1, 2, 0).numpy().astype("float32")
        # traj_maps to numpy with bs * T channels
        bs, T, old_H, old_W = input_traj_maps.shape
        input_traj_maps = (
            input_traj_maps.view(bs * T, old_H, old_W)
            .permute(1, 2, 0)
            .numpy()
            .astype("float32")
        )
        # keypoints to list of tuples
        # need to clamp because some slightly exit from the image
        abs_pixel_coord[:, :, 0] = np.clip(
            abs_pixel_coord[:, :, 0], a_min=0, a_max=old_W - 1e-3
        )
        abs_pixel_coord[:, :, 1] = np.clip(
            abs_pixel_coord[:, :, 1], a_min=0, a_max=old_H - 1e-3
        )
        keypoints = list(map(tuple, abs_pixel_coord.reshape(-1, 2)))

        transform = A.Compose(
            [
                # SAFE AUGS, flips and 90rots
                A.augmentations.transforms.HorizontalFlip(p=0.5),
                A.augmentations.transforms.VerticalFlip(p=0.5),
                A.augmentations.transforms.Transpose(p=0.5),
                A.augmentations.geometric.rotate.RandomRotate90(p=1.0),
                # HIGH RISKS - HIGH PROBABILITY OF KEYPOINTS GOING OUT
                A.OneOf(
                    [  # perspective or shear
                        A.augmentations.geometric.transforms.Perspective(
                            scale=0.05, pad_mode=cv2.BORDER_CONSTANT, p=1.0
                        ),
                        A.augmentations.geometric.transforms.Affine(
                            shear=(-10, 10), mode=cv2.BORDER_CONSTANT, p=1.0
                        ),  # shear
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [  # translate
                        A.augmentations.geometric.transforms.ShiftScaleRotate(
                            shift_limit_x=0.01,
                            shift_limit_y=0,
                            scale_limit=0,
                            rotate_limit=0,
                            border_mode=cv2.BORDER_CONSTANT,
                            p=1.0,
                        ),  # x translations
                        A.augmentations.geometric.transforms.ShiftScaleRotate(
                            shift_limit_x=0,
                            shift_limit_y=0.01,
                            scale_limit=0,
                            rotate_limit=0,
                            border_mode=cv2.BORDER_CONSTANT,
                            p=1.0,
                        ),  # y translations
                        A.augmentations.geometric.transforms.Affine(
                            translate_percent=(0, 0.01), mode=cv2.BORDER_CONSTANT, p=1.0
                        ),  # random xy translate
                    ],
                    p=0.2,
                ),
                # random rotation
                A.augmentations.geometric.rotate.Rotate(
                    limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.4
                ),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            additional_targets={"traj_map": "image", "rgb_image": "image"},
        )
        transformed = transform(
            image=image,
            keypoints=keypoints,
            traj_map=input_traj_maps,
            rgb_image=rgb_image,
        )

        # FROM NUMPY BACK TO TENSOR
        image = torch.tensor(transformed["image"]).permute(2, 0, 1)
        rgb_image = torch.tensor(transformed["rgb_image"]).permute(2, 0, 1)
        C, new_H, new_W = image.shape
        abs_pixel_coord = torch.tensor(transformed["keypoints"]).view(
            batch_data["seq_coord"].shape
        )
        input_traj_maps = (
            torch.tensor(transformed["traj_map"])
            .permute(2, 0, 1)
            .view(bs, T, new_H, new_W)
        )

        # NEW AUGMENTATION: INVERT TIME
        if random.random() > 0.5:
            abs_pixel_coord = abs_pixel_coord.flip(dims=(0,))
            input_traj_maps = input_traj_maps.flip(dims=(0,))

        if random.random() < self.goal_aug_prob:
            abs_pixel_coord[:, -1, :] = self.augment_goal(
                abs_pixel_coord[:, 7, :], abs_pixel_coord[:, -1, :], image
            )

        batch_data["scene_sem"] = image
        batch_data["scene_img"] = rgb_image
        batch_data["seq_coord"] = abs_pixel_coord
        batch_data["input_traj_maps"] = input_traj_maps

        return batch_data

    def __len__(self):
        return len(self.seq_coord)

    def __getitem__(self, index):
        out = {
            "seq_coord": self.seq_coord[index],
            "scene_img": self.scene_img[index],
            "scene_sem": self.scene_sem[index],
            "scene_id": self.scene_id[index],
            "input_traj_maps": self.input_traj_maps[index],
        }
        if self.data_augmentation:
            out = self.augment_traj_and_images(out)
        return out
