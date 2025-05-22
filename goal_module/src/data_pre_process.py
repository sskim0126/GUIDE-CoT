# -*- coding: utf-8 -*-
import os
import pickle
import random
import math

import numpy as np
import torch

from src.data_src.dataset_src.dataset_create import create_dataset
from src.data_src.experiment_src.experiment_create import create_experiment
from src.models.model_utils.cnn_big_images_utils import create_tensor_image, \
    create_CNN_inputs_loop
from src.utils import maybe_makedir


def is_legitimate_traj(traj_df, step):
    agent_id = traj_df.agent_id.values
    # check if I only have 1 agent (always the same)
    if not (agent_id[0] == agent_id).all():
        print("not same agent")
        return False
    frame_ids = traj_df.frame_id.values
    equi_spaced = np.arange(frame_ids[0], frame_ids[-1] + 1, step, dtype=int)
    # check that frame rate is evenly-spaced
    if not (frame_ids == equi_spaced).all():
        print("not equi_spaced")
        return False
    # if checks are passed
    return True


# +
class Trajectory_Data_Pre_Process(object):
    def __init__(self, args):
        self.args = args

        # Trajectories and data_batches folder
        self.data_batches_path = os.path.join(
            self.args.save_dir, 'data_batches')
        maybe_makedir(self.data_batches_path)

        # Creating batches folders and files
        self.batches_folders = {}
        self.batches_confirmation_files = {}
        for set_name in ['train', 'valid', 'test']:
            # batches folders
            self.batches_folders[set_name] = os.path.join(
                self.data_batches_path, f"{set_name}_batches")
            maybe_makedir(self.batches_folders[set_name])
            # batches confirmation file paths
            self.batches_confirmation_files[set_name] = os.path.join(
                self.data_batches_path, f"finished_{set_name}_batches.txt")

        # exit pre-processing early
        if os.path.exists(self.batches_confirmation_files["test"]):
            print('Data batches already created!\n')
            return

        print("Loading dataset and experiment ...")
        self.dataset = create_dataset(self.args.dataset)
        self.experiment = create_experiment(self.args.dataset)(
            self.args.test_set, self.args)
        print("Done.\n")

        print("Preparing data batches ...")
        self.num_batches = {}
        for set_name in ['train', 'valid', 'test']:
            if not os.path.exists(self.batches_confirmation_files[set_name]):
                self.num_batches[set_name] = 0
                print(f"\nPreparing {set_name} batches ...")
                self.create_data_batches(set_name)

        print('Data batches created!\n')

    def create_data_batches(self, set_name):
        """
        Create data batches for the DataLoader object
        """
        for scene_data in self.experiment.data[set_name]:
            # break if fast_debug
            if self.args.fast_debug and self.num_batches[set_name] >= \
                    self.args.fast_debug_num:
                break
            self.make_batches(scene_data, set_name)
            print(f"Saved a total of {self.num_batches[set_name]} {set_name} "
                  f"batches ...")

        with open(self.batches_confirmation_files[set_name], "w") as f:
            f.write(f"Number of {set_name} batches: "
                    f"{self.num_batches[set_name]}")

    def make_batches(self, scene_data, set_name):
        """
        Query the trajectories fragments and make data batches.
        Notes: Divide the fragment if there are too many people; accumulate some
        fragments if there are few people.
        """
        scene_name = scene_data["scene_name"]
        scene = self.dataset.scenes[scene_name]
        delta_frame = scene.delta_frame
        downsample_frame_rate = scene_data["downsample_frame_rate"]

        df = scene_data['raw_pixel_data']

        if set_name == 'train':
            shuffle = self.args.shuffle_train_batches
        elif set_name == 'test':
            shuffle = self.args.shuffle_test_batches
        else:
            shuffle = self.args.shuffle_test_batches
        assert scene_data["set_name"] == set_name

        fragment_list = []  # container for a batch of data (list of fragments)
        
        data = df.to_numpy()
        frames = np.unique(data[:, 0]).tolist()
        frame_data = []
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :])
        skip = downsample_frame_rate
        seq_len = 20
        num_sequences = int(math.ceil((len(frames) - seq_len + 1) / skip))
        
        for idx in range(0, num_sequences * skip, skip):
            curr_seq_data = np.concatenate(
                frame_data[idx : idx + seq_len], axis=0
            )
            peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
            curr_seq = np.zeros((len(peds_in_curr_seq), 4, seq_len))
            curr_loss_mask = np.zeros((len(peds_in_curr_seq), seq_len))
            num_peds_considered = 0
            for _, ped_id in enumerate(peds_in_curr_seq):
                curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :].astype(np.float32)
                curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                if curr_ped_seq.shape[0] != seq_len or pad_end - pad_front != seq_len:
                    continue
                curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                curr_ped_seq = curr_ped_seq
                _idx = num_peds_considered
                curr_seq[_idx, 2:4, pad_front:pad_end] = curr_ped_seq
                curr_seq[_idx, 0, pad_front:pad_end] = frames[idx:idx+20]
                curr_seq[_idx, 1, pad_front:pad_end] = ped_id
                # Linear vs Non-Linear Trajectory
                num_peds_considered += 1

            if num_peds_considered > 1:
                for ped_idx in range(num_peds_considered):
                    fragment_list.append(curr_seq[ped_idx])
        
        if shuffle:
            random.shuffle(fragment_list)

        batch_acculumator = []
        batch_ids = {
            "scene_name": scene_name,
            "starting_frames": [],
            "agent_ids": [],
            "data_file_path": scene_data["file_path"]}

        for fragment_df in fragment_list:
            # break if fast_debug
            if self.args.fast_debug and self.num_batches[set_name] >= \
                    self.args.fast_debug_num:
                break

            batch_ids["starting_frames"].append(fragment_df[0,0])
            batch_ids["agent_ids"].append(fragment_df[1,0])

            batch_acculumator.append(fragment_df[2:4, :].transpose(1, 0))

            # save batch if big enough
            if len(batch_acculumator) == self.args.batch_size:
                # create and save batch
                self.massup_batch_and_save(batch_acculumator,
                                           batch_ids, set_name)

                # reset batch_acculumator and ids for new batch
                batch_acculumator = []
                batch_ids = {
                    "scene_name": scene_name,
                    "starting_frames": [],
                    "agent_ids": [],
                    "data_file_path": scene_data["file_path"]}

        # save last (incomplete) batch if there is some fragment left
        if batch_acculumator:
            # create and save batch
            self.massup_batch_and_save(batch_acculumator,
                                       batch_ids, set_name)

    def massup_batch_and_save(self, batch_acculumator, batch_ids, set_name):
        """
        Mass up data fragments to form a batch and then save it to disk.
        From list of dataframe fragments to saved batch.
        """
        abs_pixel_coord = np.stack(batch_acculumator).transpose(1, 0, 2) # 20, B, 2
        seq_list = np.ones((abs_pixel_coord.shape[0],
                            abs_pixel_coord.shape[1]))

        data_dict = {
            "abs_pixel_coord": abs_pixel_coord,
            "seq_list": seq_list,
        }

        # add cnn maps and inputs
        data_dict = self.add_pre_computed_cnn_maps(data_dict, batch_ids)
        
        B = abs_pixel_coord.shape[1]
        direction = torch.zeros(B)
        for i in range(B):
            direction[i] = self.get_direction_type(abs_pixel_coord[:8, i, :], abs_pixel_coord[8:, i, :])
        data_dict["direction"] = direction
        
        # increase batch number count
        self.num_batches[set_name] += 1
        batch_name = os.path.join(
            self.batches_folders[set_name],
            f"{set_name}_batch" + "_" + str(
                self.num_batches[set_name]).zfill(4) + ".pkl")
        # save batch
        with open(batch_name, "wb") as f:
            pickle.dump((data_dict, batch_ids), f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def add_pre_computed_cnn_maps(self, data_dict, batch_ids):
        """
        Pre-compute CNN maps used by the goal modules and add them to data_dict
        """
        abs_pixel_coord = data_dict["abs_pixel_coord"]
        scene_name = batch_ids["scene_name"]
        scene = self.dataset.scenes[scene_name]

        # numpy semantic map from 0 to 1
        img = scene.semantic_map_pred
        
        rgb_image = scene.RGB_image

        # tensor_image : semantic map
        tensor_image = create_tensor_image(
            big_numpy_image=img,
            down_factor=self.args.down_factor)
        
        rgb_image = create_tensor_image(
            big_numpy_image=rgb_image,
            down_factor=self.args.down_factor
        )
        
        input_traj_maps = create_CNN_inputs_loop(
            batch_abs_pixel_coords=torch.tensor(abs_pixel_coord).float() /
                                self.args.down_factor,
            tensor_image=tensor_image)

        data_dict["rgb_image"] = rgb_image
        data_dict["tensor_image"] = tensor_image
        data_dict["input_traj_maps"] = input_traj_maps

        return data_dict
    
    def get_direction_type(self, obs_traj, pred_traj):
        DIRECTION_RADIUS_THRESHOLD = 30
        DIRECTION_BACKWARD_THRESHOLD = 90
        DIRECTION_FORWARD_THRESHOLD = 30
        if pred_traj is None:
            obs_len = 2
            pred_len = obs_traj.shape[0] - 1
            full_traj = obs_traj
        else:
            obs_len, pred_len = obs_traj.shape[0], pred_traj.shape[0]
            full_traj = np.concatenate([obs_traj, pred_traj], axis=0)
        full_traj_disp = full_traj[1:] - full_traj[:-1]
        # Filter out static people
        if np.linalg.norm(full_traj[obs_len] - full_traj[pred_len], ord=2, axis=-1) < DIRECTION_RADIUS_THRESHOLD:
            return 4
        # Normalize rotation
        dir = full_traj[obs_len] - full_traj[obs_len - 2]
        rot = np.arctan2(dir[1], dir[0])
        traj_rot = np.array([[np.cos(rot), np.sin(-rot)],
                            [np.sin(rot), np.cos(rot)]])
        full_traj_disp_norm = full_traj_disp @ traj_rot
        future_norm = full_traj_disp_norm[obs_len:].mean(axis=0)
        future_dir = np.arctan2(future_norm[1], future_norm[0]) * 180 / np.pi
        # Filter out moving backward people
        if future_dir < -DIRECTION_BACKWARD_THRESHOLD or future_dir > DIRECTION_BACKWARD_THRESHOLD:
            return 1
        # Filter out moving left people
        if future_dir > DIRECTION_FORWARD_THRESHOLD:
            return 2
        # Filter out moving right people
        if future_dir < -DIRECTION_FORWARD_THRESHOLD:
            return 3
        # Moving forward
        return 0
