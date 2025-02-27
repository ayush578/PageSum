# -*- coding: utf-8 -*-
"""Implementation of rejection loss

Check our paper: Learning with Rejection for Abstractive Text Summarization

"""
import torch
import torch.nn as nn


def label_smoothed_nll_loss_with_rejection(
    lprobs,
    target,
    epsilon,
    ignore_index=None,
    reduce=True,
    mask=None,
    alpha=1.0,
    unk_idx=3
):
    lprobs = torch.log_softmax(lprobs, dim=-1)
    # print("lprobs: ", lprobs.shape)
    # print("target: ", target.shape)
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    # print("target: ", target.shape)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    # print("nll_loss: ",nll_loss.shape)
    # print("smooth_loss: ",smooth_loss)
    # ================== calculate rejection loss ==================
    rej_prob = torch.exp(lprobs[:, :, unk_idx]).unsqueeze(-1)
    # print("rej_prob: ",rej_prob.shape)
    if mask is not None:
        mask = mask.unsqueeze(-1).eq(0)
        keep_prob = (1. - rej_prob).masked_fill(mask, 1.0)  # 0: non-entity
    else:
        keep_prob = 1. - rej_prob
    assert keep_prob.shape == nll_loss.shape, \
        "nll_loss: {}; keep_prob: {}".format(nll_loss.shape, keep_prob.shape)    
    # print("keep_prob: ", keep_prob)
    rej_loss = keep_prob * (nll_loss + torch.log(keep_prob))
    rej_regularizer = -alpha * torch.log(keep_prob)
    nll_loss = rej_loss + rej_regularizer

    rej_smooth_loss = keep_prob * (smooth_loss + torch.log(keep_prob))
    # print(rej_smooth_loss)
    smooth_loss = rej_smooth_loss + rej_regularizer
    # ===============================================================

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    # print(nll_loss)
    # print(smooth_loss)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    # print(loss,nll_loss,smooth_loss)
    return loss, nll_loss


class LabelSmoothedLossWithRejection(nn.Module):
    def __init__(self, rejection_alpha ,ignore_idx , epsilon=0.1):
        super().__init__()
        self.rejection_alpha = rejection_alpha
        self.unk_idx = 3
        self.eps = epsilon
        self.padding_idx = ignore_idx

    def forward(self, target, lprobs, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss, nll_loss = self.compute_loss( target, lprobs, reduce=reduce)
        return loss

    def compute_loss(self, target, lprobs, reduce=True):
        # This mask marks all entities in the summary sequence. If the mask is not None,
        # rejection loss only applies to entity tokens.
        mask = None

        loss, nll_loss = label_smoothed_nll_loss_with_rejection(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
            mask=mask,
            alpha=self.rejection_alpha,
            unk_idx=self.unk_idx,
        )
        return loss, nll_loss
