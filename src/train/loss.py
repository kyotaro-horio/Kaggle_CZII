import torch
import torch.nn as nn
from monai.networks.utils import one_hot

import warnings

class FbetaLoss(nn.Module):
    def __init__(
            self, 
            beta=1.0, 
            smooth=1e-7, 
            include_background=True, 
            to_onehot_y=False, 
            softmax=False
            ):
        """
        Initialize the F-beta loss with MONAI-like options.

        Args:
            beta (float): Weight of precision in the harmonic mean. Default is 1.0 (F1 score).
            smooth (float): Small value to avoid division by zero.
            include_background (bool): Whether to include the background class in the computation.
            to_onehot_y (bool): Whether to convert y_true to a one-hot encoding.
            softmax (bool): Whether to apply softmax to y_pred.
        """
        super(FbetaLoss, self).__init__()
        self.beta = beta
        self.smooth = smooth
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax

    def forward(self, y_pred, y_true):
        """
        Compute the F-beta loss.

        Args:
            y_pred (torch.Tensor): Predicted logits or probabilities with shape (N, C, ...).
            y_true (torch.Tensor): Ground truth labels with shape (N, ...) or (N, C, ...).

        Returns:
            torch.Tensor: Computed F-beta loss.
        """
        n_pred_ch = y_pred.shape[1]

        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn('single channel prediction, `shoftmax=True` ignored.')
            else:
                y_pred = torch.softmax(y_pred, dim=1)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn('single channel prediction, `to_onehot_y=True` ignored.')
            else:
                y_true = one_hot(y_true, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn('single channel prediction, `include_background=False` ignored.')
            else:
                y_pred = y_pred[:, 1:]
                y_true = y_true[:, 1:]

        if y_pred.shape != y_true.shape:
            raise AssertionError(f"ground truth has different shape ({y_true.shape}) from input ({y_pred.shape})")

        # Flatten tensors for easier computation
        y_pred = y_pred.contiguous().view(y_pred.shape[0], -1)
        y_true = y_true.contiguous().view(y_true.shape[0], -1)

        # Compute true positives, false positives, and false negatives
        tp = torch.sum(y_true * y_pred, dim=1)
        fp = torch.sum((1 - y_true) * y_pred, dim=1)
        fn = torch.sum(y_true * (1 - y_pred), dim=1)

        # Compute precision and recall
        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        recall = (tp + self.smooth) / (tp + fn + self.smooth)

        # Compute F-beta score
        beta_sq = self.beta ** 2
        fbeta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + self.smooth)

        # Return F-beta loss (1 - F-beta score)
        return 1 - fbeta.mean()