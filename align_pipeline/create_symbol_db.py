"""
Скрипт создает словарь с образцами изображений каждого символа.
Ключ: символ;
Значение: массив образцов написания данного символа.

ВХОДНЫЕ ДАННЫЕ:
- путь к датасету, на котором модель обучалась (символы которого будут размечаться) (аргумент командной строки 1);
- путь к чекпоинту модели (аргумент командной строки 2);
- число потоков, которые загружают данные (num_workers of DataLoader) (аргумент команднойстроки 3);
- путь к словарю, куда будет сохранен словарь (аргумент командной строки 4).

ВЫХОДНЫЕ ДАННЫЕ:
словарь с образцами символов.
"""


import h5py
import sys 
import numpy as np
import torch
from model.load_data import *
from create_dataset import default_shape
import sys
sys.path.append('..')
sys.path.append('..\\..')
from image_preprocessing.PicHandler import PicHandler, Side, view_image
from model.model import *  # чтобы можно было загрузить модель
from tqdm import tqdm

BATCH_SIZE = 32
DEFAULT_SAMPLES = 1000
DSHAPE = 250


def decode(y_pred: str) -> str:
    actual_symb = res = y_pred[0]

    for si in range(len(y_pred)):
        s = y_pred[si]
        if s == actual_symb:
            continue
        elif s == NULL_SYMB:
            actual_symb = ''
            continue

        if s != actual_symb:
            res += s
    
    return res

def append_to_dict(db_dict, k, v, sizes):
    if k not in sizes.keys():
        sizes[k] = 0
        db_dict.create_dataset(k, (DEFAULT_SAMPLES, *default_shape), compression="gzip", 
                               compression_opts=4, maxshape=(None, *default_shape))

    if sizes[k] == len(db_dict[k]):
        # нужно увеличить
       db_dict[k].resize((len(db_dict[k]) + DSHAPE, *default_shape))

    db_dict[k][sizes[k], :, :] = v
    sizes[k] += 1


def collate_dict(db_dict, sizes):
    for k in sizes.keys():
        real_size = sizes[k]
        db_dict[k].resize((real_size, *default_shape))


def create_db(db_path_name, checkpoint_path, dataloader):
    model = torch.load(checkpoint_path, map_location=device)
    model.eval()
    sizes = dict()

    with h5py.File(db_path_name, 'w') as f:

        for X, y in tqdm(dataloader):
            y_pred = model(X)
            for i in range(len(X)):
                # рассмотрим одно изображение в батче
                w = X[i].shape[1]
                scale = w / y_pred[i].shape[1]

                y_pred_str = str([num_to_char(n) for n in y_pred[i].argmax(dim=1)])
                y_true_str = str([num_to_char(n) for n in y[i][1]])

                if decode(y_pred_str) != y_true_str:
                    continue

                start_symb = 0
                end_symb = 0
                tstart = y_true_str[0]
                cnt_symb = 0
                
                while end_symb < w:
                    if tstart != y_pred_str[end_symb] and y_pred_str[end_symb] != NULL_SYMB:
                        # мы нашли границу 
                        next_width = end_symb - start_symb
                        real_width = scale * next_width
                        img = X[i][start_symb*scale: min(start_symb*scale + real_width, w - 1)] 
                        sides_to_crop = [Side.top, Side.bottom]
                        if cnt_symb == len(y_true_str) - 1:
                            # последний символ, справа нужно обрезать изображение
                            sides_to_crop.append(Side.right)
                        
                        img = (1 - PicHandler.crop_by_blank(img, sides_to_crop, blank=1)) * 255
                        view_image(img)
                        append_to_dict(f, tstart, img, sizes)

                        start_symb = end_symb
                        tstart = y_pred_str[start_symb]  # точно != NULL_SYMB          
                        cnt_symb += 1

                    end_symb += 1

        collate_dict(f, sizes)


if __name__ == '__main__':
    dataset_path, checkpoint_path, num_workers, db_path = sys.argv[1:]

    train_loader = get_loaders(BATCH_SIZE, dataset_path, num_workers)['train']

    create_db(db_path, checkpoint_path, train_loader)

