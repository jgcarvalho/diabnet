import torch
import numpy as np
from typing import Union, List


def ece_mce(preds, targets, bins=10):
    lower_bound = torch.arange(0.0, 1.0, 1.0 / bins)
    upper_bound = lower_bound + 1.0 / bins

    ece = torch.zeros(1, device=preds.device)
    mce = torch.zeros(1, device=preds.device)

    for i in range(bins):
        mask = (preds > lower_bound[i]) * (preds <= upper_bound[i])
        if torch.any(mask):
            # print(lower_bound[i], upper_bound[i], preds[mask], torch.mean(targets[mask]))
            delta = torch.abs(torch.mean(preds[mask]) - torch.mean(targets[mask]))
            ece += delta * torch.mean(mask.float())
            mce = torch.max(mce, delta)

    return ece, mce


def ece(
    targets: Union[np.ndarray, List],
    predictions: Union[np.ndarray, List],
    bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error (ECE) between target labels and predicted labels.

    Parameters
    ----------
    targets : Union[np.ndarray, List]
        A numpy.ndarray or a list with experimental target labels

    predictions : Union[np.ndarray, List]
        A numpy.ndarray or a list with predicted labels of DiabNet

    bins : int, optional
        Number of bins, by default 10

    Returns
    -------
    ece : float
        Expected calibration error

    Reference
    ---------
    On Calibration of Modern Neural Networks (https://arxiv.org/abs/1706.04599)
    """
    # Check and convert types
    if type(targets) not in [np.ndarray, list]:
        raise TypeError("`targets` must be a numpy.ndarray or a list.")
    if type(predictions) not in [np.ndarray, list]:
        raise TypeError("`predictions` must be a numpy.ndarray or a list.")
    if type(bins) not in [int]:
        raise TypeError("`bins` must be a positive integer.")
    if type(targets) == list:
        targets = np.asarray(targets)
    if type(predictions) == list:
        targets = np.asarray(predictions)
    if bins <= 0:
        raise ValueError("`bins` must be a positive integer.")

    # Create lower and upper bounds
    lower_bounds = np.arange(0.0, 1.0, 1.0 / bins)
    upper_bounds = lower_bounds + 1.0 / bins

    # Initialize ece
    ece = 0.0

    # Calculate ece
    for i in range(bins):
        mask = (predictions > lower_bounds[i]) * (predictions <= upper_bounds[i])
        if np.any(mask):
            delta = np.abs(np.mean(predictions[mask]) - np.mean(targets[mask]))
            ece += delta * np.mean(mask)

    return ece


def mce(
    targets: Union[np.ndarray, List],
    predictions: Union[np.ndarray, List],
    bins: int = 10,
):
    """Calculate Maximum Calibration Error (MCE) between target labels and predicted labels.

    Parameters
    ----------
    targets : Union[np.ndarray, List]
        A numpy.ndarray or a list with experimental target labels

    predictions : Union[np.ndarray, List]
        A numpy.ndarray or a list with predicted labels of DiabNet

    bins : int, optional
        Number of bins, by default 10

    Returns
    -------
    mce : float
        Maximum calibration error

    Reference
    ---------
    On Calibration of Modern Neural Networks (https://arxiv.org/abs/1706.04599)
    """
    # Check and convert types
    if type(targets) not in [np.ndarray, list]:
        raise TypeError("`targets` must be a numpy.ndarray or a list.")
    if type(predictions) not in [np.ndarray, list]:
        raise TypeError("`predictions` must be a numpy.ndarray or a list.")
    if type(bins) not in [int]:
        raise TypeError("`bins` must be a positive integer.")
    if type(targets) == list:
        targets = np.asarray(targets)
    if type(predictions) == list:
        targets = np.asarray(predictions)
    if bins <= 0:
        raise ValueError("`bins` must be a positive integer.")

    # Create lower and upper bounds
    lower_bounds = np.arange(0.0, 1.0, 1.0 / bins)
    upper_bounds = lower_bounds + 1.0 / bins

    # Initialize mce
    mce = 0.0

    for i in range(bins):
        mask = (predictions > lower_bounds[i]) * (predictions <= upper_bounds[i])
        if np.any(mask):
            delta = np.abs(np.mean(predictions[mask]) - np.mean(targets[mask]))
            mce = np.maximum(mce, delta)

    return mce
