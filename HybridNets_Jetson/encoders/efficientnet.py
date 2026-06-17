import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import get_model_params

from ._base import EncoderMixin


class EfficientNetEncoder(EfficientNet, EncoderMixin):
    def __init__(self, stage_idxs, out_channels, model_name, depth=5):
        blocks_args, global_params = get_model_params(model_name, override_params=None)
        super().__init__(blocks_args, global_params)

        self._stage_idxs = stage_idxs
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        del self._fc

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self._conv_stem, self._bn0, self._swish),
            self._blocks[: self._stage_idxs[0]],
            self._blocks[self._stage_idxs[0] : self._stage_idxs[1]],
            self._blocks[self._stage_idxs[1] : self._stage_idxs[2]],
            self._blocks[self._stage_idxs[2] :],
        ]

    def forward(self, x):
        stages = self.get_stages()
        block_number = 0.0
        drop_connect_rate = self._global_params.drop_connect_rate

        features = []
        for stage_index in range(self._depth + 1):
            if stage_index < 2:
                x = stages[stage_index](x)
            else:
                for module in stages[stage_index]:
                    drop_connect = drop_connect_rate * block_number / len(self._blocks)
                    block_number += 1.0
                    x = module(x, drop_connect)
            features.append(x)
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("_fc.bias", None)
        state_dict.pop("_fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


efficient_net_encoders = {
    "efficientnet-b0": {
        "encoder": EfficientNetEncoder,
        "params": {
            "out_channels": (3, 32, 24, 40, 112, 320),
            "stage_idxs": (3, 5, 9, 16),
            "model_name": "efficientnet-b0",
        },
    },
    "efficientnet-b1": {
        "encoder": EfficientNetEncoder,
        "params": {
            "out_channels": (3, 32, 24, 40, 112, 320),
            "stage_idxs": (5, 8, 16, 23),
            "model_name": "efficientnet-b1",
        },
    },
    "efficientnet-b2": {
        "encoder": EfficientNetEncoder,
        "params": {
            "out_channels": (3, 32, 24, 48, 120, 352),
            "stage_idxs": (5, 8, 16, 23),
            "model_name": "efficientnet-b2",
        },
    },
    "efficientnet-b3": {
        "encoder": EfficientNetEncoder,
        "params": {
            "out_channels": (3, 40, 32, 48, 136, 384),
            "stage_idxs": (5, 8, 18, 26),
            "model_name": "efficientnet-b3",
        },
    },
    "efficientnet-b4": {
        "encoder": EfficientNetEncoder,
        "params": {
            "out_channels": (3, 48, 32, 56, 160, 448),
            "stage_idxs": (6, 10, 22, 32),
            "model_name": "efficientnet-b4",
        },
    },
    "efficientnet-b5": {
        "encoder": EfficientNetEncoder,
        "params": {
            "out_channels": (3, 48, 40, 64, 176, 512),
            "stage_idxs": (8, 13, 27, 39),
            "model_name": "efficientnet-b5",
        },
    },
    "efficientnet-b6": {
        "encoder": EfficientNetEncoder,
        "params": {
            "out_channels": (3, 56, 40, 72, 200, 576),
            "stage_idxs": (9, 15, 31, 45),
            "model_name": "efficientnet-b6",
        },
    },
    "efficientnet-b7": {
        "encoder": EfficientNetEncoder,
        "params": {
            "out_channels": (3, 64, 48, 80, 224, 640),
            "stage_idxs": (11, 18, 38, 55),
            "model_name": "efficientnet-b7",
        },
    },
}
