from model.load_data import *
from torch import nn, tensor, Tensor


HIDDEN_SIZE = 256
num_classes = OUTPUT_DIM = len(alphabet) + 1
BIDIRECTIONAL = True
BLANK = num_classes - 1


class GatedConvolution(nn.Module):
    def __init__(self, img_shape, **conv_args):
        super(GatedConvolution, self).__init__()
        self.conv = nn.Conv2d(conv_args['out_channels'], conv_args['out_channels'], kernel_size=conv_args['kernel_size'], padding=conv_args['padding'])

    def forward(self, x):
        return torch.tanh(self.conv(x)) * x


def extended_conv_layer(img_shape, **conv_params):
    return nn.Sequential(
        *[
           nn.Conv2d(**conv_params, ),
           nn.PReLU(),
           nn.BatchNorm2d(conv_params['out_channels']),
           GatedConvolution(img_shape, **conv_params)
        ]
    )

