from utils.geometry import Rect
import numpy as np
from image_preprocessing.PicHandler import view_image
from image_preprocessing.PicHandler import PicHandler


class TextBlock:
    zone: Rect  # расположение данного слова (слитного сочетания символов) на изображении
    contents: np.ndarray  # содержимое блока -- фрагмент изображения внутри zone

    def __init__(self, zone: Rect, cont: np.ndarray):
        self.zone = zone
        self.contents = cont

    def view(self) -> None:
        view_image(PicHandler.from_zero_one(self.contents), "TextBlock: %s" % str(self.zone))
