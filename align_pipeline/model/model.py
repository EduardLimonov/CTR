"""
Создание легкой модели для распознавания текста.
Модель обучается с помощью функции потерь CTC loss и далее используется для разметки границ символов
в выборке, на которой она обучалась .
"""


from torch import tensor, Tensor 
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.load_data import *
import pytorch_lightning as pl
from torch.optim.lr_scheduler import *


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


from torch.nn.modules.dropout import Dropout2d
from torch.nn.modules.activation import Sigmoid


class RecogModel(nn.Module):
    def __init__(self):
        super(RecogModel, self).__init__()

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
                nn.LSTM(input_size=32*16, hidden_size=HIDDEN_SIZE, num_layers=1,
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


class LModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_obj = nn.CTCLoss(blank=BLANK, zero_infinity=True, reduction='mean')

    def loss_fn(self, outputs, targets):
        target_labels, target_lens = targets
        input_lens = torch.full((len(target_lens),), 128, dtype=torch.int32) 
        outputs = outputs.permute(1, 0, 2)
        return self.loss_obj(outputs, target_labels, input_lens, target_lens)
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch 
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        self.log('train_loss', loss.item(), reduce_fx='mean')
        #tb_logs = {'val_loss': loss}
        return {
            'loss': loss,
        }

    """def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_func(pred, y)
        self.log('test_loss', loss.item(), prog_bar=True, reduce_fx="mean")
        output = dict({
            'test_loss': loss,
        })
        return output"""
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], _batch_index: int) -> None:
        inputs_batch, labels_batch = batch

        outputs_batch = self(inputs_batch)
        loss = self.loss_fn(outputs_batch, labels_batch)

        self.log('val_loss', loss.item(), reduce_fx="mean")
        return {
            'val_loss': loss,
        }
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.tensor([x["val_loss"] for x in outputs]).mean()  # на самом деле уже не нужно усреднять
        tb_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tb_logs}
    
    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-3)
        sched = ReduceLROnPlateau(optimizer, verbose=True, patience=3)
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss"}
        }


