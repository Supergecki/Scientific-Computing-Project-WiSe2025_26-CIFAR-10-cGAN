import torch.nn as nn


def get_baseline_loss():
    """Returns the standard Vanilla GAN loss (BCE with Logits)"""
    return nn.BCEWithLogitsLoss()


def get_lsgan_loss():
    """ Returns the MSELoss"""
    return nn.MSELoss()
