"""
Скрипт, который выделяет из HKR dataset подвыборку требуемого размера, переводит ее в полутоновое
представление (в градациях серого), и сохраняет в формате hdf5.

ВХОДНЫЕ ДАННЫЕ:
- dataset_path (параметр командной строки 1) -- строка -- путь к датасету HKR (поддиректории -- "img", "ann");
- size (параметр командной строки 2) -- число -- размер подвыборки.
- path_and_name (параметр командной строки 3) -- строка -- путь и имя создаваемого датасета

ВЫХОДНЫЕ ДАННЫЕ:
dataset.hdf5 (файл в текущей директории) -- сформированный датасет. Его содержимое:
-- под ключем "images" -- сжатые одноканальные изображения размера default_shape;
-- под ключем "labels" -- массив соответствующих строк.

"""


import sys
sys.path.append('..')
import h5py
import numpy as np
import random
from tqdm import tqdm
from image_preprocessing.PicHandler import PicHandler
from skimage.transform import resize
import os
import json


WHITE_COLOR = 255
dataset_path = ''

default_shape = (128, 1024)
skip = {10, 11, 12, 13}


def read_json(ann_path_name: str) -> tuple[str, str]:
    # пара <расшифровка, имя файла>
    with open(ann_path_name, 'rb') as file:
        data = json.loads(file.readline())
        return data["description"], data["name"]
    

def pad(arr, new_shape):
    vertical_pad = new_shape[0] - arr.shape[0]
    horizontal_pad = new_shape[1] - arr.shape[1]
    vert_add = vertical_pad % 2

    return np.pad(arr, ((vertical_pad // 2, vertical_pad // 2 + vert_add),
                        (0, horizontal_pad)),
                  'constant', constant_values=(WHITE_COLOR, ))


def create_dataset(n_samples, filename, chunk_size):
    utf8_type = h5py.string_dtype('utf-8')
    images, labels = [], []

    with h5py.File(filename, 'w') as f:
        f.create_dataset('images', (n_samples, *default_shape), compression="gzip", compression_opts=4)
        f.create_dataset('labels', (n_samples, ), dtype=utf8_type)

        cnt = 0
        files = random.sample([f for f in os.listdir(ann_path)
                               if os.path.isfile(os.path.join(ann_path, f)) and f.split('_')[0] not in skip],
                              n_samples)

        for filename in tqdm(files):
            word, img_name = read_json(ann_path + '/' + filename)
            ph = PicHandler(img_path + '/' + img_name + '.jpg')

            arr = ph.get_image()
            if arr.shape[0] > default_shape[0]:
                arr = resize(arr, (default_shape[0], int(arr.shape[1] * default_shape[0] / arr.shape[0])))
            elif arr.shape[1] > default_shape[1]:
                arr = resize(arr, (int(arr.shape[0] * default_shape[1] / arr.shape[1]), default_shape[1]))
                
            images.append(1 - pad(arr, default_shape) / 255)  
            labels.append(word)  
            cnt += 1
            
            if cnt % chunk_size == 0 and cnt != 0 or cnt == len(files):
                f['images'][cnt-chunk_size: cnt] = images
                f['labels'][cnt-chunk_size: cnt] = labels
                images, labels = [], []
            
        print("Done")
    return 


if __name__ == '__main__':
    dataset_path = sys.argv[1]
    ann_path = dataset_path + '/ann'
    img_path = dataset_path + '/img'
    dataset_filename = sys.argv[3]

    create_dataset(int(sys.argv[2]), dataset_filename, 1000)

