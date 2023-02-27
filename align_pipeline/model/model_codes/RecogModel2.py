from model.model_codes.common import *


class RecogModel2(nn.Module):
    def __init__(self):
        super(RecogModel2, self).__init__()

        self.encoder = nn.Sequential(
            *[
                nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
                # 16x128x1024
                extended_conv_layer(img_shape=(128, 1024), in_channels=8, out_channels=16, kernel_size=3, padding=1),
                nn.Dropout2d(p=0.2),
                #extended_conv_layer(img_shape=(128, 1024), in_channels=16, out_channels=16, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2), 

                # 32x64x512
                #nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
                extended_conv_layer(img_shape=(64, 512), in_channels=16, out_channels=16, kernel_size=3, padding=1),
                #extended_conv_layer(img_shape=(64, 512), in_channels=16, out_channels=32, kernel_size=3, padding=1),
                #nn.Dropout2d(p=0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # 64x32x256
                #nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                extended_conv_layer(img_shape=(32, 256), in_channels=16, out_channels=32, kernel_size=3, padding=1),
                #extended_conv_layer(img_shape=(25, 90), in_channels=BB_OUTPUT, out_channels=BB_OUTPUT, kernel_size=3, padding=1),
                nn.Dropout2d(p=0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
              
                # -> 32x16x128
                

                # 36x45x10
                #extended_conv_layer(img_shape=(10, 45), in_channels=BB_OUTPUT, out_channels=BB_OUTPUT, kernel_size=3, padding=1),
                #nn.Conv2d(in_channels=BB_OUTPUT, out_channels=BB_OUTPUT, kernel_size=(12, 1)),
                #nn.Conv2d(in_channels=BB_OUTPUT, out_channels=BB_OUTPUT, kernel_size=(1, 1)),
                #nn.Dropout2d(p=0.2),
                # 36x45x1
                #nn.Flatten(),
                #nn.LazyLinear(out_features=HIDDEN_STATE_DIM),
                #nn.LazyLinear(out_features=NUM_OF_BB*BB_OUTPUT),
            ]
        )

        self.rnn = nn.Sequential(
            *[
                nn.GRU(input_size=32*16, hidden_size=HIDDEN_SIZE, num_layers=1,
                        bidirectional=BIDIRECTIONAL, batch_first=True, )#dropout=0.2)
            ]
        )

        self.output_decoder = nn.Sequential(
            *[
                nn.Linear(in_features=HIDDEN_SIZE * (1 + int(BIDIRECTIONAL)),
                          out_features=OUTPUT_DIM),
                #nn.Dropout(p=0.1),
                nn.LogSoftmax(dim=-1)
            ]
        ) 

        self.init_conv()

    def forward(self, x: Tensor):
        enc = self.encoder(x) # BATCH_SIZE x 32 x 16 x 128
        enc = enc.permute(0, 3, 1, 2).view(-1, 128, 16*32)

        response, (hn, cn) = self.rnn(enc)
        return self.output_decoder(response)

    def init_conv(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d) or isinstance(c, nn.Conv3d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)