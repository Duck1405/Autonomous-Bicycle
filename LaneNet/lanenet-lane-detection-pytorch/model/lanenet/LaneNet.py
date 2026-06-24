# coding: utf-8
"""
LaneNet model
https://arxiv.org/pdf/1807.01726.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lanenet.loss import DiscriminativeLoss
from model.lanenet.backbone.UNet import UNet_Encoder, UNet_Decoder
from model.lanenet.backbone.ENet import ENet_Encoder, ENet_Stage3, ENet_Decoder
from model.lanenet.backbone.deeplabv3_plus.deeplabv3plus import Deeplabv3plus_Encoder, Deeplabv3plus_Decoder



class LaneNet(nn.Module):
    def __init__(self, in_ch = 3, arch="ENet", device = None):
        super(LaneNet, self).__init__()
        # no of instances for segmentation
        self.no_of_instances = 4  # embedding dimension N (the paper uses 4).
        print("Use {} as backbone".format(arch))
        self._arch = arch
        if self._arch == 'UNet':
            self._encoder = UNet_Encoder(in_ch)

            self._decoder_binary = UNet_Decoder(2)
            self._decoder_instance = UNet_Decoder(self.no_of_instances)
        elif self._arch == 'ENet':
            # Paper (Neven et al., 2018, Sec. II-A): share only ENet stages 1
            # and 2; stage 3 and the decoder are separate for each branch.
            self._encoder = ENet_Encoder(in_ch, include_stage3=False)
            self._stage3_binary = ENet_Stage3()
            self._stage3_instance = ENet_Stage3()
            self._decoder_binary = ENet_Decoder(2)
            self._decoder_instance = ENet_Decoder(self.no_of_instances)
        elif self._arch == 'DeepLabv3+':
            self._encoder = Deeplabv3plus_Encoder()

            self._decoder_binary = Deeplabv3plus_Decoder(2)
            self._decoder_instance = Deeplabv3plus_Decoder(self.no_of_instances)
        else:
            raise("Please select right model.")

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        if self._arch == 'UNet':
            c1, c2, c3, c4, c5 = self._encoder(input_tensor)
            binary = self._decoder_binary(c1, c2, c3, c4, c5)
            instance = self._decoder_instance(c1, c2, c3, c4, c5)
        elif self._arch == 'ENet':
            c = self._encoder(input_tensor)
            binary = self._decoder_binary(self._stage3_binary(c))
            instance = self._decoder_instance(self._stage3_instance(c))
        elif self._arch == 'DeepLabv3+':
            c1, c2 = self._encoder(input_tensor)
            binary = self._decoder_binary(c1, c2)
            instance = self._decoder_instance(c1, c2)
        else:
            raise("Please select right model.")

        binary_seg_ret = torch.argmax(F.softmax(binary, dim=1), dim=1, keepdim=True)

        # pix_embedding = self.sigmoid(instance)
        pix_embedding = instance

        return {
            'instance_seg_logits': pix_embedding,
            'instance_embedding': pix_embedding,
            'binary_seg_pred': binary_seg_ret,
            'binary_seg_logits': binary
        }
