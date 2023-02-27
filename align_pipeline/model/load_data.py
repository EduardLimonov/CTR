"""
Набор функций для загрузки данных из файла, их преобразования и формирования DataLoader'а.
Скрипт используется для обучения сети на малой выборке с целью дальнейшей сегментации символов
в словах из этой выборки.
"""

import h5py
from typing import *
import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import sys
sys.path.append('..')
sys.path.append('..\\..')
from utils.WriteHelper import WriteHelper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


alphabet = [' '] + WriteHelper.trivial_alphabet + list(""",.:!?-—;""")#alphabet
NULL_SYMB = '^'


def char_to_num(char: str) -> str:
    return alphabet.index(char)


def num_to_char(num: Union[str, int]) -> str:
    if int(num) < len(alphabet):
        return alphabet[int(num)]
    return NULL_SYMB


def str_to_array(string: str, lowercase: bool = True) -> np.ndarray:
    # в датасете есть интересные символы...
    string = string.replace('…', '...').replace('Қ', 'К').replace('o', 'о')\
        .replace('H', 'Н').replace('Ө', 'О').replace('қ', 'к').replace('–', '—').replace('ө', 'о')#.lower()
    if lowercase:
        string = string.lower()
    return np.array([char_to_num(ch) for ch in string])


def get_data(data_path):
    f = h5py.File(data_path, 'r')
    images, labels = f['images'][:], f['labels'][:]
    f.close()
    return images, labels


def collate_fn(batch):
    max_len = len(max(batch, key=lambda e: len(e[1]))[1])
    labels = torch.zeros((len(batch), max_len), dtype=torch.float16)

    lens = np.array([len(e[1]) for e in batch])
    to_pad = max_len - lens

    for i in range(len(batch)):
        target = batch[i][1]
        labels[i] = torch.cat((tensor(target), torch.zeros(to_pad[i])))

    X = torch.as_tensor(np.array([e[0] for e in batch]), dtype=torch.float32)
    return X.view(-1, 1, X.shape[1], X.shape[2]), \
        (labels, torch.as_tensor(lens, dtype=torch.int32))


def get_loaders(batch_size, data_path, num_workers, lowercase_labels: bool = True):
    num_workers = int(num_workers)  # часто получаем строку
    images, labels = get_data(data_path)
    targets = [str_to_array(str(string, 'utf8'), lowercase_labels) for string in labels]

    data = list(zip(images, targets))  # сортировать список по возрастанию длин слов не нужно, это может
                                       # ухудшить результаты
    train_data, test_data = train_test_split(data, train_size=0.9, shuffle=True, random_state=42)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)

    test_loader = DataLoader(
        test_data, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)

    return {
        'train': train_loader,
        'val': test_loader
    }

