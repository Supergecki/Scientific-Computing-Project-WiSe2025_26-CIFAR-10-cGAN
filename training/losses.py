import torch
import torch.nn as nn


def get_baseline_loss():
    """Returns the standard Vanilla GAN loss (BCE with Logits)"""
    return nn.BCEWithLogitsLoss()


def get_lsgan_loss():
    """ Returns the MSELoss"""
    return nn.MSELoss()


def hinge_loss_discriminator(real_logits, fake_logits):
    """
    Hinge loss for the discriminator
    penelizes real logits < 1 and fake logits > -1.
    """
    loss_real = torch.mean(nn.ReLU()(1.0 - real_logits))
    loss_fake = torch.mean(nn.ReLU()(1.0 + fake_logits))

    return (loss_real + loss_fake) / 2

def hinge_loss_generator(fake_logits):
    """
    Hinge loss for the generator
    tries to push fake logits as high as possible
    """
    return -(torch.mean(fake_logits))


