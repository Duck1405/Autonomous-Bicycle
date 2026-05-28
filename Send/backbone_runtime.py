from __future__ import annotations

from torch import nn

from encoders import get_encoder
from hybridnets.model_runtime import BiFPN, BiFPNDecoder, SegmentationHead
from utils.constants import BINARY_MODE, MULTICLASS_MODE


class HybridNetsBackboneRuntime(nn.Module):
    def __init__(self, compound_coef=0, seg_classes=1, seg_mode=MULTICLASS_MODE):
        super().__init__()
        self.compound_coef = compound_coef
        self.seg_classes = seg_classes
        self.seg_mode = seg_mode

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        conv_channel_coef = {
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        self.bifpn = nn.Sequential(
            *[
                BiFPN(
                    self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[self.compound_coef],
                    True if repeat_index == 0 else False,
                    attention=True if self.compound_coef < 6 else False,
                    use_p8=self.compound_coef > 7,
                    onnx_export=False,
                )
                for repeat_index in range(self.fpn_cell_repeats[self.compound_coef])
            ]
        )
        self.bifpndecoder = BiFPNDecoder(pyramid_channels=self.fpn_num_filters[self.compound_coef])
        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=1 if self.seg_mode == BINARY_MODE else self.seg_classes + 1,
            activation=None,
            kernel_size=1,
            upsampling=4,
        )
        self.encoder = get_encoder(
            "efficientnet-b" + str(self.backbone_compound_coef[self.compound_coef]),
            in_channels=3,
            depth=5,
            weights=None,
        )

        self._initialize_decoder(self.bifpndecoder)
        self._initialize_head(self.segmentation_head)
        self._initialize_decoder(self.bifpn)

    def forward(self, inputs):
        p2, p3, p4, p5 = self.encoder(inputs)[-4:]
        p3, p4, p5, p6, p7 = self.bifpn((p3, p4, p5))
        decoder_outputs = self.bifpndecoder((p2, p3, p4, p5, p6, p7))
        return self.segmentation_head(decoder_outputs)

    def _initialize_decoder(self, module):
        for submodule in module.modules():
            if isinstance(submodule, nn.Conv2d):
                nn.init.kaiming_uniform_(submodule.weight, mode="fan_in", nonlinearity="relu")
                if submodule.bias is not None:
                    nn.init.constant_(submodule.bias, 0)
            elif isinstance(submodule, nn.BatchNorm2d):
                nn.init.constant_(submodule.weight, 1)
                nn.init.constant_(submodule.bias, 0)
            elif isinstance(submodule, nn.Linear):
                nn.init.xavier_uniform_(submodule.weight)
                if submodule.bias is not None:
                    nn.init.constant_(submodule.bias, 0)

    def _initialize_head(self, module):
        for submodule in module.modules():
            if isinstance(submodule, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(submodule.weight)
                if submodule.bias is not None:
                    nn.init.constant_(submodule.bias, 0)
