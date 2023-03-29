import numpy as np
import h5py
import random
from dataclasses import dataclass
from typing import *
import sys
sys.path.append('../..')
from image_preprocessing.PicHandler import view_images, PicHandler
from utils.WriteHelper import WriteHelper
from utils.TextBlock import TextBlock
from utils.geometry import Point, Rect


@dataclass
class GenParams:
    tokens: list[str]
    p_fail: float
    max_width: int 
    word_h: Callable[[], int]
    space_w: Callable[[str], int]
    line_indent: Callable[[], int]
    delta_y_next_word: Callable[[], int]
    top_left_input: Point
    white_color: int = 255


@dataclass
class GenWordOutput:
    img: np.ndarray
    labels: list[(str, int, int)]

    
@dataclass
class GenDocOutput:
    img: np.ndarray
    labels: list[Rect]


class ImgGenerator:
    dbase: h5py.File

    def __init__(self, dbase: h5py.File) -> None:
        self.dbase = dbase

    def create_sample(self, word: str, standard_height: int, white_color: int = 255) -> GenWordOutput:
        """
        Создает изображение слова, выбирая случайное написание букв из self.dbase.
        
        Возвращает GenWordOutput
        """
        list_of_symbs = []
        res = np.full((4 * standard_height, 4 * standard_height * len(word)), white_color)  # холст
        tpos = (standard_height * (WriteHelper.OUT_OF_LINE_HEIGHT + 0.1), 1)

        for symb in word:
            new_symb_image = self.dbase[symb][random.randint(0, len(self.dbase[symb]) - 1)]
            new_symb_image = PicHandler.crop_by_blank(new_symb_image)
            h, w = new_symb_image.shape

            if WriteHelper.have_script(symb, superscript=True):
                # символ имеет надстрочную часть
                # 1*h + oolh * h = shape[0] => h = shape[0] / (1 + oolh) => delta = -h_upper = -h * oolh
                delta_y = -new_symb_image.shape[0] * (1 + WriteHelper.OUT_OF_LINE_HEIGHT)
            else:
                delta_y = 0

            b, r = tpos[0] + h + delta_y, tpos[1] + w
            res[tpos[0] + delta_y: b, tpos[1]: r] = new_symb_image 
            list_of_symbs.append((symb, tpos[1], r))
            tpos = (tpos[0], tpos[1] + w)

        return GenWordOutput(res, list_of_symbs)

    def gen_document(self, gen_params: GenParams) -> GenDocOutput:
        """
        Создает изображение документа, выбирая случайное написание букв из self.dbase.
        
        Возвращает GenDocOutput
        """
        words_to_point: list[TextBlock] = []
        actual_point = start_of_this_line = gen_params.top_left_input
        for i in range(len(gen_params.tokens)):
            token = gen_params.tokens[i]
            token_img = self.create_sample(token, gen_params.word_h, gen_params.white_color)
            token_width = token_img.shape[1]
            
            if actual_point.x + token_width > gen_params.max_width:
                # перенести строку 
                actual_point = start_of_this_line = Point(
                    x = gen_params.top_left_input.x,
                    y = start_of_this_line.y + gen_params.line_indent()
                )

            token_pos = Rect(actual_point, actual_point + token_img.shape[::-1])  # сначала x, потом y
            words_to_point.append(TextBlock(token_pos, token_img))

            # если будет следующий токен:
            # добавить пробел или отступ в зависимости от следующего токена; 
            # добавить случайное смещение по вертикали
            if i < len(gen_params.tokens) - 1:
                actual_point = actual_point + Point(
                    x = gen_params.space_w(gen_params.tokens[i + 1]),
                    y = gen_params.delta_y_next_word()
                )
        
        res = np.full(
                (words_to_point[-1].zone.bottom(), max(words_to_point, key=lambda x: x.zone.right())), 
                gen_params.white_color
            )  # холст
        
        list_of_zones = []
        for text_block in words_to_point:
            z = text_block.zone
            list_of_zones.append(z)
            res[z.top(): z.bottom(), z.left(): z.right()] = text_block.contents

        return GenDocOutput(res, list_of_zones)
        


if __name__ == '__main__':
    doc = ImgGenerator(h5py.File('../symb_db.hdf5')).create_sample('привет мир', 60)
    view_images([doc.img])


