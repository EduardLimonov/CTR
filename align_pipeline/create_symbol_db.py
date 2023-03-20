# coding=utf8
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
from pytorch_lightning import LightningModule
sys.path.append('..')
sys.path.append('..\\..')
from image_preprocessing.PicHandler import PicHandler, Side, view_images
from model.model import *  # чтобы можно было загрузить модель
from tqdm import tqdm
from learn_model import *
from skimage.transform import rescale

BATCH_SIZE = 1
DEFAULT_SAMPLES = 1000
DSHAPE = 250
IMG_DOWNSCALE = 2
big_img_h = default_shape[0]
new_h = big_img_h / IMG_DOWNSCALE
SHAPE_FOR_SYMBOL = (new_h, new_h * 2) #(128, 256)

downcale_k = 1 / IMG_DOWNSCALE


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
            actual_symb = s
    
    return res.replace(NULL_SYMB, '')


def append_to_dict(db_dict, k, v, sizes):
    if k not in sizes.keys():
        sizes[k] = 0
        db_dict.create_dataset(k, (DEFAULT_SAMPLES, *SHAPE_FOR_SYMBOL), compression="gzip", 
                               compression_opts=4, maxshape=(None, *default_shape))

    if sizes[k] == len(db_dict[k]):
        # нужно увеличить
       db_dict[k].resize((len(db_dict[k]) + DSHAPE, *default_shape))

    v = rescale(v, downcale_k)

    db_dict[k][sizes[k], : v.shape[0], : v.shape[1]] = v
    sizes[k] += 1


def collate_dict(db_dict, sizes):
    for k in sizes.keys():
        real_size = sizes[k]
        db_dict[k].resize((real_size, *default_shape))


def improve_symb_borders(symbol_widths: list[tuple[str, int]], word_width: int, 
                         max_increase=10, max_decrease=5, begin_w=1, end_w=2) \
                            -> list[tuple[int, int]]:

    # возвращает список границ для каждого символа
    # мы считаем, что сеть +- правильно нашла центры областей символов, но мб
    # не очень точно определила границы каждого символа; уточним их на основе 
    # априорной информации о ширине символов
    bw, ew = begin_w / (begin_w + end_w), end_w / (begin_w + end_w)
    res = []

    aprior_precision = WriteHelper.get_symb_widths(
        [s for s, w in symbol_widths], word_width
    )
    
    tstart = 0
    for (symb, width), aprior_width in zip(symbol_widths, aprior_precision):
        end = tstart + width
        delta = aprior_width - width
        modify = 0
        if delta > 0:
            modify = min(delta, max_increase)
        elif delta < 0:
            modify = max(delta, -max_decrease) 
        
        res.append(
            (
                int(max(tstart - modify * bw, 0)), 
                int(min(end + modify * ew, word_width - 1))
            )
        )
        tstart = end

    return res


def create_db(db_path_name, checkpoint_path, dataloader, empty_space_threshold: float = 0.07):
    model = LModel.load_from_checkpoint(checkpoint_path, 
        map_location=device, model=RecogModel3())
    model.eval()
    sizes = dict()

    with h5py.File(db_path_name, 'w') as f:

        for X, y in tqdm(dataloader):
            y_pred = model(X.to(device))

            for i in range(len(X)):
                # рассмотрим одно изображение в батче
                w = X[i].shape[-1]
                scale = w / y_pred[i].shape[0]
                
                y_pred_str = ''.join([num_to_char(n) for n in y_pred[i].argmax(dim=1)])
                y_true_str = ''.join([num_to_char(n) for n in y[0][i][: y[1][i]]])

                if decode(y_pred_str) != y_true_str.lower():
                    continue

                start_symb = 0
                end_symb = 0
                tstart = y_true_str[0].lower()
                cnt_symb = 0
                
                symbol_widths = []

                img_hor_cropped = PicHandler.crop_by_blank(
                    X[i, 0].cpu().numpy(), blank_line=0, blank_delta=empty_space_threshold
                )

                sum_w = 0
                while end_symb < len(y_pred_str):
                    if tstart != y_pred_str[end_symb] and (y_pred_str[end_symb] != NULL_SYMB or end_symb == len(y_pred_str) - 1):
                        # мы нашли границу 
                        next_width = end_symb - start_symb
                        real_width = min(scale * next_width, 
                                        img_hor_cropped.shape[-1] - sum_w)
                        symbol_widths.append((tstart, real_width))
                        sum_w += real_width

                        start_symb = end_symb
                        tstart = y_pred_str[start_symb]  # точно != NULL_SYMB          
                        cnt_symb += 1

                    end_symb += 1
                
                # имеем массив symbol_widths -- предположительная ширина каждого 
                # символа в слове
                borders = improve_symb_borders(
                    symbol_widths, img_hor_cropped.shape[1],
                    max_increase = 10, max_decrease = 12
                )
                for symb, (l, r) in zip(y_true_str, borders):
                    if WriteHelper.is_space(symb):
                        continue
                    img = X[i, 0, :, l: r].cpu().numpy()
                    try:
                        crop_res = PicHandler.crop_by_blank(img, [Side.top, Side.bottom], blank_line=0, blank_delta=empty_space_threshold)
                    except:
                        # в окно по какой-то причине попало пустое пространство
                        continue
                    img = (1 - crop_res) * 255
                    if min(crop_res.shape) <= 4: 
                        continue
                    append_to_dict(f, WriteHelper.to_hdf5_key(symb), img, sizes)

        collate_dict(f, sizes)


if __name__ == '__main__':
    dataset_path, checkpoint_path, num_workers, db_path = sys.argv[1:]

    train_loader = get_loaders(BATCH_SIZE, dataset_path, num_workers)['train']

    create_db(db_path, checkpoint_path, train_loader)

