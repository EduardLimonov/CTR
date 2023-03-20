from typing import *
import pickle

PATH_TO_MEAN_WIDTHS = '../resource/mean_symb_width'


class WriteHelper:
    SUPERSCRIPT: Set[str] = {s for s in 'бвёйАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'}
    SUBSCRIPT: Set[str] = {s for s in 'дзруф'}
    alphabet: List[str] = [char for char in """АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"""]
    trivial_alphabet: List[str] = [char for char in """абвгдеёжзийклмнопрстуфхцчшщъыьэюя"""]
    punctuation: List[str] = [char for char in ",.;!?"]
    
    OUT_OF_LINE_HEIGHT: float = 0.9

    with open(PATH_TO_MEAN_WIDTHS, 'rb') as f:
        widths = pickle.load(f)

    @staticmethod
    def __have_script(s: str, charset: Set[str]) -> bool:
        for char in s:
            if char in charset:
                return True
        return False

    @staticmethod
    def have_script(s: str, superscript: bool) -> bool:
        if superscript:
            return WriteHelper.__have_script(s, WriteHelper.SUPERSCRIPT)
        else:
            return WriteHelper.__have_script(s, WriteHelper.SUBSCRIPT)

    @staticmethod
    def char_to_num(char: str) -> str:
        return str(WriteHelper.alphabet.index(char))

    @staticmethod
    def num_to_char(num: Union[str, int]) -> str:
        return WriteHelper.alphabet[int(num)]

    @staticmethod
    def is_punctuation(char: str) -> bool:
        return char in WriteHelper.punctuation

    @staticmethod
    def has_punctuation(s: str) -> bool:
        for ps in WriteHelper.punctuation:
            if ps in s:
                return True

        return False

    @staticmethod
    def has_spaces(s: str) -> bool:
        return ' ' in s

    @staticmethod
    def get_symb_widths(symbols: List[str], word_width: int) -> List[int]:
        symb_w_relative = [WriteHelper.__get_symb_w(s) * WriteHelper.__get_coef(s) for s in symbols]
        word_w_relative = sum(symb_w_relative)
        scale = word_width / word_w_relative
        symb_w = [int(sw * scale) for sw in symb_w_relative]
        return symb_w


    @staticmethod
    def __get_coef(s: str) -> float:
        if WriteHelper.have_script(s, True) or WriteHelper.have_script(s, False):
            return 1.7  # TODO : remove magic numbers
        else:
            return 1

    @staticmethod
    def __get_symb_w(symb: str) -> float:
        if symb in WriteHelper.widths.keys():
            return WriteHelper.widths[symb]
        else:
            return 0.05  # пунктуация; TODO : remove magic numbers

    @staticmethod
    def is_space(symb: str) -> bool:
        return symb in (' ', '\t', '\n')

    @staticmethod
    def to_hdf5_key(symb) -> str:
        if symb in WriteHelper.punctuation:
            return '<%s>' % symb
        else:
            return symb

