import torch
import torch.nn as nn


class H_Net(nn.Module):
    """H-Net (Neven et al., 2018, "Towards End-to-End Lane Detection", Table I).

    Input : an RGB image scaled to 128x64 (width x height), i.e. a tensor of
            shape (N, 3, 64, 128) in PyTorch's (N, C, H, W) layout.
    Output: 6 homography coefficients per image, shape (N, 6) = [a, b, c, d, e, f].

    The 6 numbers assemble into the constrained perspective-transform matrix

        H = [[a, b, c],
             [0, d, e],
             [0, f, 1]]

    (see build_H below). The structural zeros and the fixed bottom-right 1 are
    what reduce a full 8-DOF homography to the paper's 6 DOF and keep horizontal
    lines horizontal under the transform.
    """

    def __init__(self):
        super().__init__()

        def conv_block(in_ch, out_ch):
            # 3x3, stride 1, padding 1 -> spatial size is preserved (the
            # downsampling is done only by the MaxPool layers), then BN + ReLU.
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_block(3, 16),
            conv_block(16, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 128x64 -> 64x32
            conv_block(16, 32),
            conv_block(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 64x32  -> 32x16
            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 32x16  -> 16x8
        )

        # After 3 pools on a 128x64 input: 64 channels x 8 x 16 = 8192 features.
        self.regressor = nn.Sequential(
            nn.Linear(64 * 8 * 16, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 6),                      # the 6 homography params
        )

        # Start from the identity homography (a=d=1, b=c=e=f=0 -> H = I): zero
        # the final layer's weights and set its bias accordingly. This is the
        # Spatial-Transformer-style initialization that keeps the perspective
        # division (f*y + 1) well-conditioned at the start of training; the head
        # "unlocks" once the final layer gets its first gradient update.
        nn.init.zeros_(self.regressor[-1].weight)
        with torch.no_grad():
            self.regressor[-1].bias.copy_(
                torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
            )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.regressor(x)


def build_H(params):
    """Assemble (N, 6) coefficients [a, b, c, d, e, f] into (N, 3, 3) matrices:

        H = [[a, b, c],
             [0, d, e],
             [0, f, 1]]
    """
    a, b, c, d, e, f = params.unbind(dim=1)
    zeros = torch.zeros_like(a)
    ones = torch.ones_like(a)
    return torch.stack([
        torch.stack([a, b, c], dim=1),
        torch.stack([zeros, d, e], dim=1),
        torch.stack([zeros, f, ones], dim=1),
    ], dim=1)


if __name__ == "__main__":
    model = H_Net()
    dummy = torch.randn(2, 3, 64, 128)   # (N, C, H, W) for a 128x64 (WxH) image
    params = model(dummy)
    H = build_H(params)
    print("params shape:", tuple(params.shape))   # (2, 6)
    print("H shape:     ", tuple(H.shape))         # (2, 3, 3)
    print("H[0]:\n", H[0])
