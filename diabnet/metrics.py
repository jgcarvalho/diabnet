import torch
import numpy as np
from typing import Union, Tuple


def ece_mce(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    bins: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate Expected Calibration Error (ECE) and Maximum Calibration Error
    (MCE) between predicted labels and target labels.

    Parameters
    ----------
    predictions : torch.Tensor
        A torch.ndarray or a list with predicted labels of DiabNet
    targets : torch.Tensor
        A numpy.ndarray or a list with experimental target labels
    bins : int, optional
        Number of bins, by default 10

    Returns
    -------
    ece : torch.Tensor
        Expected calibration error
    mce : torch.Tensor
        Maximum calibration error

    Reference
    ---------
    On Calibration of Modern Neural Networks (https://arxiv.org/abs/1706.04599)
    """
    # Check and convert types
    if type(predictions) not in [torch.Tensor]:
        raise TypeError("`predictions` must be a torch.Tensor.")
    if type(targets) not in [torch.Tensor]:
        raise TypeError("`targets` must be a torch.Tensor.")
    if type(bins) not in [int]:
        raise TypeError("`bins` must be a positive integer.")
    if bins <= 0:
        raise ValueError("`bins` must be a positive integer.")

    # Create lower and upper bounds
    lower_bound = torch.arange(0.0, 1.0, 1.0 / bins)
    upper_bound = lower_bound + 1.0 / bins

    # Initialize ece and mce
    ece = torch.zeros(1, device=predictions.device)
    mce = torch.zeros(1, device=predictions.device)

    # Calculate ece and mce
    for i in range(bins):
        mask = (predictions > lower_bound[i]) * (predictions <= upper_bound[i])
        if torch.any(mask):
            delta = torch.abs(torch.mean(predictions[mask]) - torch.mean(targets[mask]))
            ece += delta * torch.mean(mask.float())
            mce = torch.max(mce, delta)

    return ece, mce


# def ece(
#     targets: Union[torch.Tensor, np.ndarray],
#     predictions: Union[torch.Tensor, np.ndarray],
#     bins: int = 10,
# ) -> torch.Tensor:
#     """Calculate Expected Calibration Error (ECE) between target labels and
#     predicted labels.

#     Parameters
#     ----------
#     targets : Union[torch.Tensor, np.ndarray]
#         A torch.Tensor or a numpy.ndarray with experimental target labels
#     predictions : Union[torch.Tensor, np.ndarray]
#         A torch.Tensor or a numpy.ndarray with predicted labels of DiabNet
#     bins : int, optional
#         Number of bins, by default 10

#     Returns
#     -------
#     ece : torch.Tensor
#         Expected calibration error

#     Reference
#     ---------
#     On Calibration of Modern Neural Networks (https://arxiv.org/abs/1706.04599)
#     """
#     # Check and convert types
#     if type(targets) not in [torch.Tensor, np.ndarray]:
#         raise TypeError("`targets` must be a torch.Tensor or a numpy.ndarray.")
#     if type(predictions) not in [torch.Tensor, np.ndarray]:
#         raise TypeError("`predictions` must be a torch.Tensor or a numpy.ndarray.")
#     if type(bins) not in [int]:
#         raise TypeError("`bins` must be a positive integer.")
#     if type(targets) == torch.Tensor:
#         targets = targets.detach().cpu().numpy()
#     if type(predictions) == torch.Tensor:
#         predictions = predictions.detach().cpu().numpy()
#     if bins <= 0:
#         raise ValueError("`bins` must be a positive integer.")

#     # Create lower and upper bounds
#     lower_bounds = np.arange(0.0, 1.0, 1.0 / bins)
#     upper_bounds = lower_bounds + 1.0 / bins

#     # Initialize ece
#     ece = 0.0

#     # Calculate ece
#     for i in range(bins):
#         mask = (predictions > lower_bounds[i]) * (predictions <= upper_bounds[i])
#         if np.any(mask):
#             delta = np.abs(np.mean(predictions[mask]) - np.mean(targets[mask]))
#             ece += delta * np.mean(mask)

#     return torch.tensor(ece)


# def mce(
#     targets: Union[torch.Tensor, np.ndarray],
#     predictions: Union[torch.Tensor, np.ndarray],
#     bins: int = 10,
# ) -> torch.Tensor:
#     """Calculate Maximum Calibration Error (MCE) between target labels and
#     predicted labels.

#     Parameters
#     ----------
#     targets : Union[torch.Tensor, np.ndarray]
#         A torch.Tensor or a numpy.ndarray with experimental target labels
#     predictions : Union[torch.Tensor, np.ndarray]
#         A torch.Tensor or a numpy.ndarray with predicted labels of DiabNet
#     bins : int, optional
#         Number of bins, by default 10

#     Returns
#     -------
#     mce : torch.Tensor
#         Maximum calibration error

#     Reference
#     ---------
#     On Calibration of Modern Neural Networks (https://arxiv.org/abs/1706.04599)
#     """
#     # Check and convert types
#     if type(targets) not in [torch.Tensor, np.ndarray]:
#         raise TypeError("`targets` must be a torch.Tensor or a numpy.ndarray.")
#     if type(predictions) not in [torch.Tensor, np.ndarray]:
#         raise TypeError("`predictions` must be a torch.Tensor or a numpy.ndarray.")
#     if type(bins) not in [int]:
#         raise TypeError("`bins` must be a positive integer.")
#     if type(targets) == torch.Tensor:
#         targets = targets.detach().cpu().numpy()
#     if type(predictions) == torch.Tensor:
#         predictions = predictions.detach().cpu().numpy()
#     if bins <= 0:
#         raise ValueError("`bins` must be a positive integer.")

#     # Create lower and upper bounds
#     lower_bounds = np.arange(0.0, 1.0, 1.0 / bins)
#     upper_bounds = lower_bounds + 1.0 / bins

#     # Initialize mce
#     mce = 0.0

#     for i in range(bins):
#         mask = (predictions > lower_bounds[i]) * (predictions <= upper_bounds[i])
#         if np.any(mask):
#             delta = np.abs(np.mean(predictions[mask]) - np.mean(targets[mask]))
#             mce = np.maximum(mce, delta)

#     return torch.tensor(mce)
