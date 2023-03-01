"""
Создание легкой модели для распознавания текста.
Модель обучается с помощью функции потерь CTC loss и далее используется для разметки границ символов
в выборке, на которой она обучалась .
"""

from model.model_codes.common import *
from torch import tensor, Tensor 
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.load_data import *
import pytorch_lightning as pl
from torch.optim.lr_scheduler import *


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
        sched = ReduceLROnPlateau(optimizer, verbose=True, patience=4)
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss"}
        }


