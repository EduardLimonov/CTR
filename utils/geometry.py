from __future__ import annotations
from typing import *
from math import sqrt


def len_of_intersection(a1: int, a2: int, b1: int, b2: int) -> int:
    '''
    длина пересечения двух отрезков на прямой

    :param a1: начало отрезка a
    :param a2: конец отрезка a
    :param b1: начало отрезка b
    :param b2: конец отрезка b
    :return: длина пересечения отрезков a, b
    '''

    if not(b1 <= a1 < b2 or b1 < a2 <= b2 or a1 <= b1 < a2 or a1 < b2 <= a2):
        # пересечения нет или это одна точка
        return 0

    start, stop = max(a1, b1), min(a2, b2)
    return stop - start


def sign(x: float) -> float:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def distance(x1, y1, x2, y2) -> float:
    dx, dy = x2 - x1, y2 - y1
    return sqrt(dx * dx + dy * dy)


class Point:
    x: int
    y: int

    def __init__(self, x: int, y: int):
        self.x, self.y = x, y

    def __str__(self) -> str:
        return '(x: %d, y: %d)' % (self.x, self.y)

    def __add__(self, other: Point | tuple[int, int]) -> Point:
        if type(other) == tuple:
            other = Point(*other)
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, k: float) -> Point:
        return Point(int(self.x * k), int(self.y * k))

    def __eq__(self, other: Point) -> bool:
        return self.x == other.x and self.y == other.y

    @staticmethod
    def distance(a: Point, b: Point) -> float:
        dp = a - b
        return sqrt(dp.x * dp.x + dp.y * dp.y)


class Rect:
    top_left: Point
    bottom_right: Point

    def __init__(self, tl: Point, br: Point):
        self.top_left, self.bottom_right = tl, br

    def __str__(self) -> str:
        return 'top-left: %s; right-bottom: %s' % (self.top_left, self.bottom_right)

    def __contains__(self, item: Point) -> bool:
        return self.top_left.x <= item.x <= self.bottom_right.x and \
               self.top_left.y <= item.y <= self.bottom_right.y

    def __eq__(self, other: Rect) -> bool:
        return self.top_left == other.top_left and self.bottom_right == other.bottom_right

    def shape(self) -> Tuple[int, int]:
        return self.w(), self.h()

    def center(self) -> Point:
        return self.top_left + Point(int(self.w()/2), int(self.h()/2))

    def w(self) -> int:
        return self.bottom_right.x - self.top_left.x

    def h(self) -> int:
        return self.bottom_right.y - self.top_left.y

    @staticmethod
    def intersects(a: Rect, b: Rect) -> bool:
        return a.intersects_x(b) and a.intersects_y(b)

    @staticmethod
    def distance(a: Rect, b: Rect) -> float:
        if Rect.intersects(a, b):
            return 0

        ac, bc = a.center(), b.center()
        dc = ac - bc
        # non_inters = dc.x > a.w() + b.w() and dc.y > a.h() + b.h()  # ни одна из проекций на оси x, y не пересекается
        ix, iy = a.intersects_x(b), a.intersects_y(b)

        if not ix and not iy:
            # кратчайшее расстояние -- между вершинами
            x1, y1 = ac.x - a.w() / 2 * sign(dc.x), ac.y - a.h() / 2 * sign(dc.y)
            x2, y2 = bc.x + b.w() / 2 * sign(dc.x), bc.y + b.h() / 2 * sign(dc.y)
            return distance(x1, y1, x2, y2)

        elif ix:
            # достаточно найти расстояние по y
            y1 = ac.y - a.h() / 2 * sign(dc.y)
            y2 = bc.y + b.h() / 2 * sign(dc.y)
            return abs(y1 - y2)

        elif iy:
            # достаточно найти расстояние по x
            x1 = ac.x - a.w() / 2 * sign(dc.x)
            x2 = bc.x + b.w() / 2 * sign(dc.x)
            return abs(x1 - x2)

    def left(self) -> int:
        return self.top_left.x

    def right(self) -> int:
        return self.bottom_right.x

    def top(self) -> int:
        return self.top_left.y

    def bottom(self) -> int:
        return self.bottom_right.y

    def is_on_left(self, other: Rect) -> bool:
        return self.right() <= other.left()

    def is_on_right(self, other: Rect) -> bool:
        return other.is_on_left(self)

    def is_on_top(self, other: Rect) -> bool:
        return self.bottom() <= other.top()

    def is_on_bottom(self, other: Rect) -> bool:
        return other.is_on_top(self)

    def intersects_x(self, other: Rect) -> bool:
        return self.left() <= other.left() <= self.right() or \
               self.left() <= other.right() <= self.right() or \
               other.left() <= self.left() <= other.right() or \
               other.left() <= self.right() <= other.right()

    def intersects_y(self, other: Rect) -> bool:
        return self.top() <= other.top() <= self.bottom() or \
               self.top() <= other.bottom() <= self.bottom() or \
               other.top() <= self.top() <= other.bottom() or \
               other.top() <= self.bottom() <= other.bottom()

    def area(self) -> int:
        return self.w() * self.h()

    def distance_to_y(self, y: int) -> int:
        return min(abs(self.top() - y), abs(self.bottom() - y))

    def multiply_coordinates(self, px: float, py: float) -> None:
        self.top_left = Point(int(self.left() * px), int(self.top() * py))
        self.bottom_right = Point(int(self.right() * px), int(self.bottom() * py))

    def move(self, dx: int, dy: int) -> None:
        self.top_left.x += dx
        self.top_left.y += dy
        self.bottom_right.x += dx
        self.bottom_right.y += dy

    def limit_bottom_right(self, max_x: int, max_y: int) -> None:
        self.bottom_right.x = min(max_x, self.right())
        self.bottom_right.y = min(max_y, self.bottom())

    def scale_x(self, scale_k: float) -> None:
        dx = int(self.w() * (1 - scale_k) / 2)
        self.top_left.x -= dx
        self.bottom_right.x += dx

    def scale_y(self, scale_k: float) -> None:
        dy = int(self.h() * (1 - scale_k) / 2)
        self.top_left.y -= dy
        self.bottom_right.y += dy

    def scale_xy(self, scale_x: float, scale_y: float) -> None:
        self.scale_x(scale_x)
        self.scale_y(scale_y)

    def scale(self, scale_k: float) -> None:
        self.scale_x(scale_k)
        self.scale_y(scale_k)

    def intersection(self, other: Rect) -> Optional[Rect]:
        if not Rect.intersects(self, other):
            return None
        else:
            l = max(self.left(), other.left())
            t = max(self.top(), other.top())
            b = min(self.bottom(), other.bottom())
            r = min(self.right(), other.right())
            return Rect(Point(l, t), Point(r, b))

    def union(self, other: Rect) -> Rect:
        l = min(self.left(), other.left())
        t = min(self.top(), other.top())
        b = max(self.bottom(), other.bottom())
        r = max(self.right(), other.right())
        return Rect(Point(l, t), Point(r, b))

    @staticmethod
    def union_of_rects(rects: List[Rect]) -> Rect:
        res = rects[0]
        for i in range(1, len(rects)):
            res = res.union(rects[i])

        return res

    def iou_val(self, other: Rect) -> float:
        if not Rect.intersects(self, other):
            return 0
        elif self == other:
            return True

        intersection = self.intersection(other)
        return intersection.area() / (self.area() + other.area() - intersection.area())


if __name__ == '__main__':
    print(len_of_intersection(1, 4, -2, 3))
