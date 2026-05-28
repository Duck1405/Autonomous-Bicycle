import torch.nn as nn

from . import _utils as utils


class EncoderMixin:
    @property
    def out_channels(self):
        return self._out_channels[: self._depth + 1]

    def set_in_channels(self, in_channels, pretrained=True):
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        utils.patch_first_conv(model=self, new_in_channels=in_channels, pretrained=pretrained)

    def get_stages(self):
        raise NotImplementedError

    def make_dilated(self, output_stride):
        raise ValueError(
            "Dilated encoders are not packaged in this runtime bundle. "
            f"Requested output_stride={output_stride}."
        )
