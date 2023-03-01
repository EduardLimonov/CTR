"""
Обучение нейронной сети

ВХОДНЫЕ ДАННЫЕ (аргументы командной строки):
- путь к датасету;
- путь для сохранения чекпоинтов;
- число потоков, которые загружают данные (num_workers of DataLoader)

ВЫХОДНЫЕ ДАННЫЕ:
чекпоинты и логи обучения модели.
"""

import pytorch_lightning as pl
from model.model import *
from model.load_data import *
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model.model_codes.RecogModel import RecogModel
from model.model_codes.RecogModel2 import RecogModel2
from model.model_codes.RecogModel3 import RecogModel3
import sys
#sys.path.append('..\\')

if __name__ == '__main__':
    loaders = get_loaders(batch_size=32, data_path=sys.argv[1], num_workers=int(sys.argv[3]))

    models = [
        LModel(RecogModel()), 
        LModel(RecogModel2()), 
        LModel(RecogModel3())
    ]
    #logger = TensorBoardLogger("tb_logs", name="rec_model")
    

    for model in models:
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.005,
                                            patience=6, verbose=True, mode="min")
        trainer = pl.Trainer(
            devices=1, accelerator="auto",
            max_epochs=80,
            default_root_dir=sys.argv[2],
            callbacks=[early_stop_callback]
            #logger=logger
        )
        trainer.fit(model, loaders['train'], loaders['val'])

