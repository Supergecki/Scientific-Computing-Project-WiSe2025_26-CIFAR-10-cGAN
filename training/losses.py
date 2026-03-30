import torch
import torch.nn as nn


def get_baseline_loss():
    return nn.BCEWithLogitsLoss()


def hinge_loss_discriminator(real_logits, fake_logits):
    loss_real = torch.mean(nn.ReLU()(1.0 - real_logits))
    loss_fake = torch.mean(nn.ReLU()(1.0 + fake_logits))
    return loss_real + loss_fake


def hinge_loss_generator(fake_logits):
    return -torch.mean(fake_logits)
