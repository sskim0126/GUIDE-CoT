import torch
import torch.nn as nn

from src.losses import Goal_BCE_loss
from src.data_src.dataset_src.dataset_create import create_dataset
from src.metrics import FDE_best_of_goal, FDE_best_of_goal_world
from src.models.model_utils.U_net_CNN import VisSemUNet
from src.models.model_utils.sampling_2D_map import sampling, \
    TTST_test_time_sampling_trick



class VisSem(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.dataset = create_dataset(self.args.dataset)

        ##################
        # MODEL PARAMETERS
        ##################

        # GOAL MODULE PARAMETERS
        self.num_image_channels = 6

        # U-net encoder channels
        self.enc_chs = (self.num_image_channels + 8, 64, 128, 256, 512, 1024)
        # U-net decoder channels
        self.dec_chs = (1024, 512, 256, 128, 64)


        ##################
        # MODEL LAYERS
        ##################
        self.goal_module = VisSemUNet(
            enc_chs=self.enc_chs,
            dec_chs=self.dec_chs,
            out_chs=1)

    def prepare_inputs(self, batch_data, batch_id):
        """
        Prepare inputs to be fed to a generic model.
        """
        # we need to remove first dimension which is added by torch.DataLoader
        # float is needed to convert to 32bit float
        selected_inputs = {k: v.squeeze(0).float().to(self.device) if \
            torch.is_tensor(v) else v for k, v in batch_data.items()}
        # extract seq_list
        seq_list = selected_inputs["seq_list"]
        # decide which is ground truth
        ground_truth = selected_inputs["abs_pixel_coord"] # (seq_length, N, 2)

        scene_name = batch_id["scene_name"][0]
        scene = self.dataset.scenes[scene_name]
        selected_inputs["scene"] = scene

        return selected_inputs, ground_truth, seq_list

    def init_losses(self):
        losses = {
            "goal_BCE_loss": 0,
        }
        return losses

    def set_losses_coeffs(self):
        losses_coeffs = {
            "goal_BCE_loss": 1,
        }
        return losses_coeffs
    
    def init_sample_losses(self):
        losses = self.init_losses()
        sample_losses = {k: [] for k in losses.keys()}
        return sample_losses

    def init_train_metrics(self):
        train_metrics = {
            # "ADE": [],
            "FDE": [],
        }
        return train_metrics

    def init_test_metrics(self):
        test_metrics = {
            "FDE": [],
            "FDE_world": [],
        }
        return test_metrics

    def init_best_metrics(self):
        best_metrics = {
            "FDE": 1e9,
            "FDE_world": 1e9,
            "goal_BCE_loss": 1e9,
        }
        return best_metrics

    def best_valid_metric(self):
        return "FDE"

    def compute_loss_mask(self, seq_list, obs_length: int = 8):
        """
        Get a mask to denote whether to account predictions during loss
        computation. It is supposed to calculate losses for a person at
        time t only if his data exists from time 0 to time t.

        Parameters
        ----------
        seq_list : PyTorch tensor
            input is seq_list[1:]. Size = (seq_len,N_pedestrians). Boolean mask
            that is =1 if pedestrian i is present at time-step t.
        obs_length : int
            number of observation time-steps

        Returns
        -------
        loss_mask : PyTorch tensor
            Shape: (seq_len,N_pedestrians)
            loss_mask[t,i] = 1 if pedestrian i if present from beginning till time t
        """
        loss_mask = seq_list.cumprod(dim=0)
        # we should not compute losses for step 0, as ground_truth and
        # predictions are always equal there
        # loss_mask[0] = 0
        return loss_mask

    def compute_model_losses(self,
                             outputs,
                             ground_truth,
                             loss_mask,
                             inputs,
                             aux_outputs):
        """
        Compute loss for a generic model.
        """
        assert outputs is None, "This model does not predict trajectories!"
        
        out_maps_GT_goal = inputs["input_traj_maps"][:, -1]
        goal_logit_map = aux_outputs["goal_logit_map"]
        goal_BCE_loss = Goal_BCE_loss(
            goal_logit_map, out_maps_GT_goal, loss_mask)

        losses = {
            "goal_BCE_loss": goal_BCE_loss,
        }

        return losses
    def compute_model_metrics(self,
                              metric_name,
                              phase,
                              predictions,
                              ground_truth,
                              metric_mask,
                              all_aux_outputs,
                              inputs,
                              obs_length=8):
        """
        Compute model metrics for a generic model.
        Return a list of floats (the given metric values computed on the batch)
        """
        assert predictions is None, "This model does not predict trajectories!"

        # scale back to original dimension
        ground_truth = ground_truth.detach() * self.args.down_factor

        # convert to world coordinates
        scene = inputs["scene"]

        GT_world = scene.make_world_coord_torch(ground_truth)

        if metric_name == 'FDE':
            return FDE_best_of_goal(all_aux_outputs, ground_truth,
                                    metric_mask, self.args)
        elif metric_name == 'FDE_world':
            return FDE_best_of_goal_world(all_aux_outputs, scene,
                                          GT_world, metric_mask, self.args)
        else:
            raise ValueError("This metric has not been implemented yet!")

    def forward(self, inputs, num_samples=1, if_test=False):
        ##################
        # PREDICT GOAL
        ##################
        # extract precomputed map for goal goal_idx
        _, num_agents, _ = inputs["abs_pixel_coord"].shape
        tensor_image = inputs["tensor_image"]
        obs_traj_maps = inputs["input_traj_maps"][:, :self.args.obs_length]
        input_goal_module = torch.cat((tensor_image, obs_traj_maps), dim=1)
        rgb_image = inputs["rgb_image"]

        # compute goal maps
        goal_logit_map_start = self.goal_module(input_goal_module, rgb_image)
                
        goal_prob_map = torch.sigmoid(
            goal_logit_map_start / self.args.sampler_temperature)

        if self.args.use_ttst and num_samples > 1:
            LENGTH_THRESHOLD = 20
            static = inputs["lengths"] < LENGTH_THRESHOLD
            goal_point_start = TTST_test_time_sampling_trick(
                goal_prob_map,
                num_goals=num_samples,
                device=self.device,
                static=static,
                start_position=inputs["abs_pixel_coord"][0],
                last_position=inputs["abs_pixel_coord"][self.args.obs_length-1])
            goal_point_start = goal_point_start.squeeze(2).permute(1, 0, 2)
        else:
            goal_point_start = sampling(goal_prob_map, num_samples=num_samples)
            goal_point_start = goal_point_start.squeeze(1) # B, 1, 2
        
        # START SAMPLES LOOP
        all_outputs = None
        all_aux_outputs = []
        for sample_idx in range(num_samples):
            goal_point = goal_point_start[:, sample_idx]
            aux_outputs = {
                "goal_logit_map": goal_logit_map_start,
                "goal_point": goal_point
            }
            all_aux_outputs.append(aux_outputs)

        # from list of dict to dict of list (and then tensors)
        all_aux_outputs = {k: torch.stack([d[k] for d in all_aux_outputs])
                           for k in all_aux_outputs[0].keys()}
        return all_outputs, all_aux_outputs
