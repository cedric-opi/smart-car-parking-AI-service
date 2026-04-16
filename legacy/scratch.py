import heapq
import math
from typing import Tuple


def astar(
    grid_rows: int,
    grid_cols: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    turn_penalty: float = 2.0,
):
    sc, sr = start
    gc, gr = goal

    def h(c, r):
        return abs(c - gc) + abs(r - gr)

    open_heap = []
    heapq.heappush(open_heap, (h(sc, sr), 0, sc, sr, 0, 0))
    came_from = {(sc, sr, 0, 0): None}
    g_score = {(sc, sr, 0, 0): 0.0}

    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]

    while open_heap:
        _, g, c, r, pdc, pdr = heapq.heappop(open_heap)

        if (c, r) == (gc, gr):
            path = []
            cur = (gc, gr, pdc, pdr)
            while cur is not None:
                path.append((cur[0], cur[1]))
                cur = came_from[cur]
            path.reverse()
            return path

        for dc, dr in directions:
            nc, nr = c + dc, r + dr
            if not (0 <= nc < grid_cols and 0 <= nr < grid_rows):
                continue

            step_cost = math.sqrt(dc * dc + dr * dr)

            penalty = 0
            if (pdc, pdr) != (0, 0) and (pdc, pdr) != (dc, dr):
                penalty = turn_penalty

            ng = g + step_cost + penalty
            state = (nc, nr, dc, dr)
            if ng < g_score.get(state, float("inf")):
                g_score[state] = ng
                came_from[state] = (c, r, pdc, pdr)
                heapq.heappush(open_heap, (ng + h(nc, nr), ng, nc, nr, dc, dr))

    return []


print(astar(10, 10, (0, 0), (9, 9)))
