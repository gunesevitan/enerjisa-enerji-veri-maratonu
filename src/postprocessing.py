import numpy as np


def clip_negative_values(predictions):

    """
    Clip negative predictions to 0

    Parameters
    ----------
    predictions [array-like of shape (n_samples)]: Predictions array

    Returns
    -------
    predictions [array-like of shape (n_samples)]: Clipped predictions array
    """

    predictions = np.clip(predictions, a_min=0, a_max=None)
    return predictions


def clip_night_values(predictions, night_mask):

    """
    Clip predictions at night to 0

    Parameters
    ----------
    predictions [array-like of shape (n_samples)]: Predictions array
    night_mask [array-like of shape (n_samples)]: Boolean mask array of night

    Returns
    -------
    predictions [array-like of shape (n_samples)]: Clipped predictions array
    """

    predictions[night_mask] = 0
    return predictions
