import torch.nn as nn
import torch
import torch.nn.functional as F
import layerfilter as filter_layer
import ltr.models.layers.activation as activation
from ltr.models.layers.distance import DistanceMap
import math



class DiMPL2SteepestDescentGN(nn.Module):
    """A simpler optimizer module that uses L2 loss.
    args:
        num_iter:  Number of default optimization iterations.
        feat_stride:  The stride of the input feature.
        init_step_length:  Initial scaling of the step length (which is then learned).
        gauss_sigma:  The standard deviation of the label function.
        hinge_threshold:  Threshold for the hinge-based loss (see DiMP paper).
        init_filter_reg:  Initial filter regularization weight (which is then learned).
        min_filter_reg:  Enforce a minimum value on the regularization (helps stability sometimes).
        detach_length:  Detach the filter every n-th iteration. Default is to never detech, i.e. 'Inf'.
        alpha_eps:  Term in the denominator of the steepest descent that stabalizes learning.
    """
    def __init__(self, num_iter=1, feat_stride=16, init_step_length=1.0, gauss_sigma=1.0, hinge_threshold=-999,
                 init_filter_reg=1e-2, min_filter_reg=1e-3, detach_length=float('Inf'), alpha_eps=0.0):
        super().__init__()

        self.num_iter = num_iter
        self.detach_length = detach_length
        

    def forward(self, weights, feat, label, num_iter=None, compute_losses=True):
        """Runs the optimizer module.
        Note that [] denotes an optional dimension.
        args:
            weights:  Initial weights. Dims (sequences, feat_dim, wH, wW).
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            sample_weight:  Optional weight for each sample. Dims: (images_in_sequence, [sequences]).
            num_iter:  Number of iterations to run.
            compute_losses:  Whether to compute the (train) loss in each iteration.
        returns:
            weights:  The final oprimized weights.
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""

        # Sizes
        num_iter = self.num_iter if num_iter is None else num_iter
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = (weights.shape[-2], weights.shape[-1])
        output_sz = (feat.shape[-2] + (weights.shape[-2] + 1) % 2, feat.shape[-1] + (weights.shape[-1] + 1) % 2)



        weight_iterates = [weights]
        losses = []

        for i in range(num_iter):
            if i > 0 and i % self.detach_length == 0:
                weights = weights.detach()

            # Compute residuals
            scores = filter_layer.apply_filter(feat, weights)
            residuals = (scores - label_map)

            if compute_losses:
                losses.append(((residuals**2).sum())/num_sequences)

            # Compute gradient
            residuals_mapped = (residuals)
            weights_grad = filter_layer.apply_feat_transpose(feat, residuals_mapped, filter_sz, training=self.training)

            # Map the gradient with the Jacobian
            scores_grad = filter_layer.apply_filter(feat, weights_grad)
            

            # Compute optimal step length
            alpha_num = (weights_grad * weights_grad).sum(dim=(1,2,3))
            alpha_den = ((scores_grad * scores_grad).reshape(num_images, num_sequences, -1).sum(dim=(0,2)) + (reg_weight + self.alpha_eps) * alpha_num).clamp(1e-8)
            alpha = alpha_num / alpha_den

            # Update filter
            weights = weights - (alpha.reshape(-1, 1, 1, 1)) * weights_grad

            # Add the weight iterate
            weight_iterates.append(weights)

        if compute_losses:
            scores = filter_layer.apply_filter(feat, weights)
            losses.append((((scores - label_map)**2).sum())/num_sequences)

        return weights, weight_iterates, losses

