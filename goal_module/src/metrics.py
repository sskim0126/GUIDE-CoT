import torch
import numpy as np
from scipy.stats import gaussian_kde


def compute_metric_mask(seq_list):
    """
    Get a mask to denote whether to account predictions during metrics
    computation. It is supposed to calculate metrics only for pedestrians
    fully present during observation and prediction time-steps.

    Parameters
    ----------
    seq_list : PyTorch tensor
        Size = (seq_len,N_pedestrians).
        Boolean mask that is =1 if pedestrian i is present at time-step t.

    Returns
    -------
    metric_mask : PyTorch tensor
        Shape: (N_pedestrians,)
        metric_mask[i] = 1 if pedestrian i if fully present during
        observation and prediction time-steps.
    """
    metric_mask = seq_list.cumprod(dim=0)
    # fully present on the whole seq_length interval
    metric_mask = metric_mask[-1] > 0
    return metric_mask


def check_metrics_inputs(predictions,
                         ground_truth,
                         metric_mask):
    num_sample, seq_length, N_agents, num_coords = predictions.shape
    # assert data shape
    assert len(predictions.shape) == 4, \
        f"Expected 4D (MxTxNxC) array for predictions, got {predictions.shape}"
    assert ground_truth.shape == (seq_length, N_agents, num_coords), \
        f"Expected 3D (TxNxC) array for ground_truth, got {ground_truth.shape}"
    assert metric_mask.shape == (N_agents,), \
        f"Expected 1D (N) array for metric_mask, got {metric_mask.shape}"

    # assert all data is valid
    assert torch.isfinite(predictions).all(), \
        "Invalid value found in predictions"
    assert torch.isfinite(ground_truth).all(), \
        "Invalid value found in ground_truth"
    assert torch.isfinite(metric_mask).all(), \
        "Invalid value found in metric_mask"


def FDE_best_of(predictions: torch.Tensor,
                ground_truth: torch.Tensor,
                metric_mask: torch.Tensor,
                obs_length: int = 8) -> tuple:
    """
    Compute FDE metric - Best-of-K selected.
    The best FDE from the samples is selected, based on the best ADE.
    Torch implementation. Returns a list of floats, one for each full
    example in the batch.

    Parameters
    ----------
    predictions : torch.Tensor
        The output trajectories of the prediction model.
        Shape: num_sample * seq_len * N_pedestrians * (x,y)
    ground_truth : torch.Tensor
        The target trajectories.
        Shape: seq_len * N_pedestrians * (x,y)
    metric_mask : torch.Tensor
        Mask to denote if pedestrians are fully present.
        Shape: (N_pedestrians,)
    obs_length : int
        Number of observation time-steps

    Returns
    ----------
    FDE : list of float
        Final displacement error
    """
    check_metrics_inputs(predictions, ground_truth, metric_mask)

    # l2-norm for each time-step
    error = torch.norm(predictions - ground_truth, p=2, dim=3)
    # only calculate for fully present pedestrians
    error_full = error[:, -1, metric_mask]

    # best error over samples
    final_error, _ = error_full.min(dim=0)

    return final_error.tolist()


def FDE_best_of_goal(all_aux_outputs: list,
                     ground_truth: torch.Tensor,
                     metric_mask: torch.Tensor,
                     args,
                     ) -> list:
    """
    Compute the best of 20 FDE metric between the final position GT and
    the predicted goal.
    Works with a goal architecture model only.
    Returns a list of float, FDE errors for each full pedestrians in
    the batch.
    """
    # take only last temporal step (final destination)
    ground_truth = ground_truth[-1]
    end_point_pred = all_aux_outputs["goal_point"]
    end_point_pred = end_point_pred.to(ground_truth.device) * args.down_factor

    # difference
    FDE_error = ((end_point_pred - ground_truth)**2).sum(-1) ** 0.5

    # take minimum over samples
    # take only agents with full trajectories
    best_error_full, _ = FDE_error[:, metric_mask].min(dim=0)

    return best_error_full.flatten().cpu().tolist()


def FDE_best_of_goal_world(all_aux_outputs: list,
                           scene,
                           ground_truth: torch.Tensor,
                           metric_mask: torch.Tensor,
                           args,
                           ) -> list:
    """
    Compute the best of 20 FDE metric between the final position GT and
    the predicted goal.
    Returns a list of float, FDE errors for each full pedestrians in
    the batch.
    """
    # take only last temporal step (final destination)
    ground_truth = ground_truth[-1]
    end_point_pred = all_aux_outputs["goal_point"]
    end_point_pred = end_point_pred.to(ground_truth.device) * args.down_factor
    # from pixel to world coordinates
    end_point_pred = scene.make_world_coord_torch(end_point_pred)

    # difference
    FDE_error = ((end_point_pred - ground_truth)**2).sum(-1) ** 0.5

    # take minimum over samples
    # take only agents with full trajectories
    best_error_full, _ = FDE_error[:, metric_mask].min(dim=0)

    return best_error_full.flatten().cpu().tolist()
