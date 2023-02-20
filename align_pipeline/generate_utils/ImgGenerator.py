import numpy as np
import h5py
import random
from dataclasses import dataclass
import sys
sys.path.append('../..')
from utils.WriteHelper import WriteHelper
from utils.geometry import Point


@dataclass
class GenParams:
    p_fail: float
    max_width: int 
    word_h: callable[int]
    space_w: callable[int]
    line_indent: callable[int]
    delta_y_next_word: callable[int]
    top_left_input: Point


@dataclass
class GenOutput:
    img: np.ndarray
    labels: list[(str, int, int)]


class ImgGenerator:
    dbase: h5py.File

    def __init__(self, dbase: h5py.File) -> None:
        self.dbase = dbase

    def create_sample(self, word: str, standard_height: int, white_color: int = 255) -> GenOutput:
        """
        Создает изображение слова, выбирая случайное написание букв из self.dbase.
        
        Возвращает GenOutput
        """
        list_of_symbs = []
        res = np.full((4 * standard_height, 4 * standard_height * len(word)), white_color)  # холст
        tpos = (standard_height * (WriteHelper.OUT_OF_LINE_HEIGHT + 0.1), 1)

        for symb in word:
            new_symb_image = self.dbase[symb][random.randint(0, len(self.dbase[symb]) - 1)]
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

        return GenOutput(res, list_of_symbs)

    def gen_document(self, gen_params: GenParams) -> GenOutput:
        """
        Создает изображение документа, выбирая случайное написание букв из self.dbase.
        
        Возвращает GenOutput
        """
        pass 


if __name__ == '__main__':
    def test1():
        pass


