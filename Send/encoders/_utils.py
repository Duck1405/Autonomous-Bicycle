import torch
import torch.nn as nn


def patch_first_conv(model, new_in_channels, default_in_channels=3, pretrained=True):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            break

    weight = module.weight.detach()
    module.in_channels = new_in_channels

    if not pretrained:
        module.weight = nn.parameter.Parameter(
            torch.Tensor(
                module.out_channels,
                new_in_channels // module.groups,
                *module.kernel_size,
            )
        )
        module.reset_parameters()
    elif new_in_channels == 1:
        module.weight = nn.parameter.Parameter(weight.sum(1, keepdim=True))
    else:
        new_weight = torch.Tensor(
            module.out_channels,
            new_in_channels // module.groups,
            *module.kernel_size,
        )
        for index in range(new_in_channels):
            new_weight[:, index] = weight[:, index % default_in_channels]
        new_weight = new_weight * (default_in_channels / new_in_channels)
        module.weight = nn.parameter.Parameter(new_weight)
