"""
Функционал, используемый для обработки изображения.

Функции view_image, view_images являются отладочными или используются для демонстрации работы алгоритмов.

Класс PicHandler используется для перевода изображение в полутоновое представление (в градациях серого),
для его фильтрации, бинаризации (алгоритмы  адаптивной бинаризации), обрезки и т.д.
"""


from __future__ import annotations
from typing import *
import cv2
import numpy as np
from pythreshold.utils import *
from utils.geometry import Rect
from skimage.transform import resize, rescale
import enum


class FilterType(enum.Enum):
    GAUSSIAN_FILTER = 0
    MEDIAN_FILTER = 1

class StackingType(enum.Enum):
    HORIZONTAL = 0
    VERTICAL = 1


class Side(enum.Enum):
    left = 'left'
    right = 'right'
    top = 'top'
    bottom = 'bottom'
    all = 'all'


def view_image(image: np.ndarray, name_of_window: str = 'Image'):
    # выводит на экран изображение image (массив представления BGR)
    cv2.namedWindow(name_of_window, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def view_images(images: Iterable[np.ndarray], name_of_window: str = 'Images', 
                stacking: StackingType = StackingType.HORIZONTAL):
    # выводит на экран набор изображений images (массивы представлений BGR)
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    res = images[0]
    if stacking == StackingType.HORIZONTAL:
        ax = 1
    else:
        ax = 0
    for i in range(1, len(images)):
        res = np.concatenate((res, images[i]), axis=ax)

    cv2.imshow(name_of_window, res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class PicHandler:
    img: np.ndarray  # объект изображения в BGR

    def __init__(self, image: Union[str, np.ndarray], make_copy: bool = True, is_colored: bool = True):
        # image -- путь к файлу с изображением или np.ndarray -- представление изображения BGR в виде массива;
        # если передан массив, то make_copy: bool -- необходимо ли работать с копией переданного массива;
        # если передан путь к файлу, то изображение открывается, и при is_colored = True делается черно-белым

        if isinstance(image, np.ndarray):
            if make_copy:
                self.img = image.copy()
            else:
                self.img = image
            if is_colored:
                self.img = self.make_black_and_white(self.img)

        elif isinstance(image, type('')):
            t = cv2.imread(image)

            if isinstance(t, type(None)):
                # изображение не удалось загрузить
                raise Exception("Некорректный путь к изображению")

            if is_colored:
                t = self.make_black_and_white(t)

            self.img = t

    def get_image(self) -> np.ndarray:
        return self.img

    @staticmethod
    def make_black_and_white(img: np.ndarray) -> np.ndarray:
        # возвращает черно-белое изображение, соответствующее цветному img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def apply_filter(self, filter_type: FilterType, filter_size: int = 9) -> None:
        # модифицирует self.img, применяя к нему фильтр, соответствующий значению filter_type

        def apply_gaussian(img, figure_size=9):
            return cv2.GaussianBlur(img, (figure_size, figure_size), 0)

        def apply_median(img, figure_size=9):
            return cv2.medianBlur(img, figure_size)

        if filter_type == FilterType.GAUSSIAN_FILTER:
            self.img = apply_gaussian(self.img, filter_size)
        elif filter_type == FilterType.MEDIAN_FILTER:
            self.img = apply_median(self.img, filter_size)

    def apply_global_bin_filter(self, thresh: int = 220) -> None:
        # во все пикселы, значения которых больше порога thresh, устанавливаются значения 255
        # во все остальные -- 0
        mask = self.img >= thresh
        self.img[mask] = 255
        self.img[~mask] = 0

    def apply_adaptive_bin_filter(self, mode: int = 0, **params):
        if mode == 0:
            self.img = apply_threshold(self.img, bradley_roth_threshold(self.img, **params))
        else:
            self.img = apply_threshold(self.img, singh_threshold(self.img))

    def get_copy(self) -> np.ndarray:
        return self.img.copy()

    def show(self):
        # выводит на экран изображение
        view_image(self.img, 'pic_handler_image')

    @staticmethod
    def crop(img: np.ndarray, rect: Rect, make_copy: bool = False) -> np.ndarray:
        min_x, max_x = rect.left(), rect.right()
        min_y, max_y = rect.top(), rect.bottom()

        res = img[min_y: max_y + 1, min_x: max_x + 1]
        return res.copy() if make_copy else res

    @staticmethod
    def draw_pixels(rect: Rect, pixels: Set[Tuple[int, int]]) -> np.ndarray:
        # имеем pixels -- координаты закрашенных "1" точек на исходном изображении.
        # Метод рисует то, что должно быть в rect
        sh = rect.shape()
        dy, dx = -rect.top(), -rect.left()
        res = np.zeros((sh[1] + 1, sh[0] + 1), dtype=np.uint8)
        for x, y in pixels:
            res[y + dy, x + dx] = 1

        return res

    def make_zero_one(self) -> np.ndarray:
        # возвращает матрицу для бинарного изображения: 1, если пиксел не закрашен, иначе 0
        # Не вызывайте этот метод, если изображение не бинаризовано
        return (self.img == 0).astype(np.uint8)

    @staticmethod
    def from_zero_one(mat: np.ndarray) -> np.ndarray:
        # mat: 1, если пиксел не закрашен, иначе 0
        return mat * 255

    def draw_rect(self, rect: Rect, color: int = 0) -> None:
        left, right, top, bottom = rect.left(), rect.right(), rect.top(), rect.bottom()
        for x_static in (left, right):
            for y_dyn in range(top, bottom + 1):
                self.img[y_dyn, x_static] = color

        for y_static in (top, bottom):
            for x_dyn in range(left, right + 1):
                self.img[y_static, x_dyn] = color

    def __rebin(self) -> None:
        self.apply_global_bin_filter()

    def resize(self, shape: Tuple[int, int]) -> None:
        self.img = resize(self.img, shape, preserve_range=True)
        self.__rebin()

    def rescale(self, scale: float) -> None:
        self.img = rescale(self.img, scale)
        self.__rebin()

    def exec_pipeline(self, pipeline: Callable, make_copy: bool=False, **params) -> PicHandler:
        if make_copy:
            ph = PicHandler(self.img)
        else:
            ph = self
        pipeline(ph, **params)
        return self

    @staticmethod
    def crop_by_blank(img, side: Side | list[Side] = Side.all, blank_line: int=255):
        x_min, y_min, y_max, x_max = 0, 0, *img.shape

        if type(side) == list:
            if len(side) == 1:
                side = side[0]

        if type(side) != list:
            if side == Side.all:
                side = [s for s in Side if s != Side.all]
            else:
                side = [side]

        for s in side:
            """
            match s:
                case Side.left: x_min = np.where(np.all(img != blank_line, axis=0))[0]
                case Side.right: x_max = np.where(np.all(img != blank_line, axis=0))[-1]
                case Side.left: y_min = np.where(np.all(img != blank_line, axis=1))[0]
                case Side.left: x_max = np.where(np.all(img != blank_line, axis=1))[-1]"""
            
            
            if s == Side.left: x_min = np.where(np.all(img != blank_line, axis=0))[0]
            elif s == Side.right: x_max = np.where(np.all(img != blank_line, axis=0))[-1]
            elif s == Side.left: y_min = np.where(np.all(img != blank_line, axis=1))[0]
            elif s == Side.left: x_max = np.where(np.all(img != blank_line, axis=1))[-1]

        return img[y_min: y_max, x_min: x_max]


if __name__ == '__main__':

    fname = 'ex1.jpg'
    f1_scan = '../test/handwritten1.jpg'
    f1_photo = '../test/hand1.jpg'
    f2_scan = '../test/handwritten2.jpg'
    f2_photo = '../test/hand2.jpg'

    def test_filers():
        fname = f2_photo
        size = 9
        p1_scan_med, p1_scan_gaus = PicHandler(fname), PicHandler(fname)
        img = p1_scan_med.get_copy()
        p1_scan_med.apply_filter(MEDIAN_FILTER, size)
        p1_scan_gaus.apply_filter(GAUSSIAN_FILTER, size)
        view_images([img, p1_scan_med.img, p1_scan_gaus.img])


    def test_bin():
        p1_scan, p1_photo = PicHandler(f1_scan), PicHandler(f1_photo)
        img_s = p1_scan.get_copy()
        img_p = p1_photo.get_copy()

        p1_scan.apply_adaptive_bin_filter()
        p1_photo.apply_adaptive_bin_filter()

        view_images([img_s, p1_scan.img])
        view_images([img_p, p1_photo.img])


    def test_bin_advanced(fname):
        p1_photo = PicHandler(fname)
        p1_low, p1_high = PicHandler(fname), PicHandler(fname)
        p1_low.apply_adaptive_bin_filter(w=0.05, w_size=25)
        p1_photo.apply_adaptive_bin_filter(w=0.1, w_size=25)
        p1_high.apply_adaptive_bin_filter(w=0.15, w_size=25)

        view_images([p1_low.img, p1_photo.img, p1_high.img])


    def test_filter_bin(fname, **params):
        size = 5
        p1_scan_gaus = PicHandler(fname)
        p1_filter_bin = PicHandler(fname)

        p1_filter_bin.apply_filter(GAUSSIAN_FILTER, size)
        p1_filter_bin.apply_adaptive_bin_filter(**params)

        p1_scan_gaus.apply_adaptive_bin_filter(**params)
        p1_scan_gaus.apply_filter(GAUSSIAN_FILTER, size)
        p1_scan_gaus.apply_adaptive_bin_filter(**params)
        view_images([p1_filter_bin.img, p1_scan_gaus.img])


    test_bin()
    '''test_filter_bin(fname=f1_scan)
    test_filter_bin(fname=f1_photo, w=0.1, w_size=20)
    test_filter_bin(fname=f2_scan, w=0.2, w_size=20)
    test_filter_bin(fname=f2_photo, w=0.08, w_size=20)'''


    def fixed_filters_test():
        thresh = 200
        for filter_size in range(1, 8, 2):
            print(filter_size)
            phg, phm = PicHandler(fname), PicHandler(fname)
            pgd = PicHandler(fname)
            default = phg.get_copy()

            phg.apply_filter(GAUSSIAN_FILTER, filter_size)
            phm.apply_filter(MEDIAN_FILTER, filter_size)

            g, m = phg.get_copy(), phm.get_copy()

            phg.apply_fixed_bin_filter(thresh)
            phm.apply_fixed_bin_filter(thresh)
            pgd.apply_fixed_bin_filter()

            print(pgd.get_copy().shape)
            view_images((pgd.get_copy(), phg.get_copy(), phm.get_copy())) #((default, g, m, pgd.get_copy(), phg.get_copy(), phm.get_copy()))

    def adaptive_filters_test():
        ph1, ph2 = PicHandler(fname), PicHandler(fname)
        default = ph1.get_copy()

        ph1.apply_adaptive_bin_filter(0)
        #ph2.apply_adaptive_bin_filter(1)

        view_images((default, ph1.get_copy(), ))#ph2.get_copy()))

    def adaptive_and_noise_filters_test():
        ph1, ph2, ph3 = PicHandler(fname), PicHandler(fname), PicHandler(fname)
        default = ph1.get_copy()
        fsize = 3

        ph1.apply_filter(MEDIAN_FILTER, fsize)
        ph1.apply_adaptive_bin_filter()

        ph2.apply_filter(GAUSSIAN_FILTER, fsize)
        ph2.apply_adaptive_bin_filter()

        ph3.apply_filter(MEDIAN_FILTER, fsize)
        ph3.apply_fixed_bin_filter(200)

        view_images((ph1.get_copy(), ph2.get_copy(), ph3.get_copy()), )#stacking=VERTICAL)

        a = ph3.make_zero_one().sum(axis=1)
        print(a)

