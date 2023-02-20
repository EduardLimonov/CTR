from collections import deque
from utils.geometry import Rect
from typing import *
import numpy as np


def get_connect_components(adj_lists: Dict[Any, List[Any]]) -> List[Set[Any]]:
    """
    Имеем списки смежности, помещенные в словарь; ключ -- вершина, значение -- список смежных с ней вершин
    возвращает список всех компонент связности в данном графе

    :param adj_lists: множество списков смежности
    :return: список компонент связности в графе
    """

    visited = set()
    comps = []

    for v in adj_lists.keys():
        if v in visited:
            continue
        new_comp = {v}
        visited.add(v)
        should_visit = deque(adj_lists[v])
        while len(should_visit):  # обойдем все смежные с v вершины
            u = should_visit.popleft()
            new_comp.add(u)
            visited.add(u)
            for t in adj_lists[u]:
                if t not in visited and t not in should_visit:
                    # мы еще не заходили в t ни при обходе компоненты, ни до этого
                    should_visit.append(t)

        comps.append(new_comp)

    return comps


def unite(components: List[Set[int]], to_unite: List[Tuple[int, int]]) -> List[Set[int]]:
    def unite_sets(sets: Iterable[Set]) -> Set:
        # объединяет все множества в наборе
        ans = set()
        for s in sets:
            ans = ans.union(s)

        return ans

    adj_list = dict()

    # для каждого номера компоненты определим все другие компоненты, с которыми мы будем ее объединять
    for u1, u2 in to_unite:
        for u in (u1, u2):
            if u not in adj_list.keys():
                adj_list[u] = []

        if u2 not in adj_list[u1]:
            adj_list[u1].append(u2)
        if u1 not in adj_list[u2]:
            adj_list[u2].append(u1)

    supercomponents: List[Set[int]] = get_connect_components(adj_list)  # список множеств; каждое множество --
    # номера компонент связности, которые нужно будет объединить между собой

    const_comp = [i for i in range(len(components)) if i not in adj_list.keys()]

    return [
               unite_sets(
                   components[i] for i in supercomponent
               )
               for supercomponent in supercomponents
           ] + [components[j] for j in const_comp]
    # наконец-то объединяем множества пикселов; индексы объединяемых множеств содержатся в supercomponents
    # множества const_comp оставляем неизменными и просто добавляем в ответ


def find_max_by_order(s: Set[Tuple[int, int]], axis: int, order_function: Callable[[int, int], bool]) -> int:
    # order_function(a, b) == True => a -- max по отношению порядка order_function
    res = None
    for point in s:
        if isinstance(res, type(None)):
            res = point[axis]
            continue

        t = point[axis]
        if order_function(t, res):
            res = t

    return res


'''def postprocess_segmentation(results: List, criterion: Callable[[Any, Any], bool],
                             union_function: Callable = None, max_iter: int = -1) -> List:
    if union_function is None:
        union_function = lambda a, b: a.union(b)

    while True:
        changed = False
        i = 0
        while i < len(results):
            cnt = 0
            j = i + 1
            while j < len(results):
                if i == j:
                    continue
                if criterion(results[i], results[j]):
                    results[i] = union_function(results[i], results[j])
                    results.pop(j)
                    changed = True
                    cnt = 0
                else:
                    j += 1
                    cnt += 1
                    if changed and cnt >= max_iter >= 0:
                        # для i-го элемента выполнили объединение; дальнейших объединений давно не было
                        break
            i += 1

        if not changed:
            break
    return results'''


def postprocess_segmentation(results: List, criterion: Callable[[Any, Any, float], bool], local_scope_sens: float,
                             union_function: Callable = None, distance_metric: Callable = None) -> List:
    if union_function is None:
        union_function = lambda a, b: a.union(b)

    if distance_metric is None:
        distance_metric = lambda a, b: Rect.distance(a, b)

    actual_results = np.ones((len(results),), dtype=np.bool)
    distances = np.zeros((len(results), len(results)))
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            if i == j:
                continue
            distances[i, j] = distances[j, i] = distance_metric(results[i], results[j])

    while True:
        changed = False
        for i in np.where(actual_results)[0]:
            # actual_results[i] != 0  <=>  results[i] != 0
            if not actual_results[i]:
                continue
            need_to_check = np.where((distances[i] <= local_scope_sens) & actual_results)[0]
            # индексы, которые нужно просмотреть

            for j in need_to_check:
                if i == j:
                    continue
                if criterion(results[i], results[j], distances[i, j]):
                    results[i] = union_function(results[i], results[j])
                    results[j] = None
                    # обновляем расстояния, т.к. теперь i, j -- эквивалентны
                    distances[i] = distances[j] = np.minimum(distances[i], distances[j])
                    distances[:, i] = distances[:, j] = distances[i]
                    actual_results[j] = False

                    changed = True
                else:
                    j += 1
            i += 1

        if not changed:
            break
    return [r for r in results if not (r is None)]
