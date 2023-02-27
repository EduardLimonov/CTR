from typing import *


class WriteHelper:
    SUPERSCRIPT: Set[str] = {s for s in 'бвёйАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'}
    SUBSCRIPT: Set[str] = {s for s in 'дзруф'}
    alphabet: List[str] = [char for char in """АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"""]
    trivial_alphabet: List[str] = [char for char in """абвгдеёжзийклмнопрстуфхцчшщъыьэюя"""]
    punctuation: List[str] = [char for char in ",.;!?"]
    
    OUT_OF_LINE_HEIGHT: float = 0.9

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
