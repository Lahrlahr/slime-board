from collections import defaultdict, deque
import numpy as np
import random


def make_pos2straight():
    def lcp(a, b):
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def add_graph(graph, seq):
        for key in list(graph.keys()):
            common = lcp(key, seq)
            if common == 0:
                continue
            if common == len(key):
                add_graph(graph[key], seq[common:])
                return
            if common == len(seq):
                new_node = {key[common:]: graph[key]}
                graph[seq] = new_node
                del graph[key]
                return

            new_node = {
                key[common:]: graph[key],
                seq[common:]: {}
            }
            graph[key[:common]] = new_node
            del graph[key]
            return
        if seq:
            graph[seq] = {}

    def add_seqs(graph, seqs):
        for seq in seqs:
            for i, a in enumerate(seq):
                left = tuple(seq[:i][::-1])
                right = tuple(seq[i + 1:])

                add_graph(graph[a], left)
                add_graph(graph[a], right)

    posmap = defaultdict(dict)
    add_seqs(posmap,
             [[0, 1, 2, 3, 4], [20, 21, 22, 23, 24], [30, 31, 32, 33, 34], [50, 51, 52, 53, 54], [60, 61, 62, 63, 64],
              [80, 81, 82, 83, 84], [90, 91, 92, 93, 94], [110, 111, 112, 113, 114],
              [20, 15, 10, 5, 0, 126, 127, 120, 64, 69, 74, 79, 84],
              [24, 19, 14, 9, 4, 124, 123, 122, 60, 65, 70, 75, 80],
              [50, 45, 40, 35, 30, 124, 125, 126, 94, 99, 104, 109, 114],
              [54, 49, 44, 39, 34, 122, 121, 120, 90, 95, 100, 105, 110]
                 , [20, 15, 10, 5, 0, 94, 99, 104, 109, 114], [24, 19, 14, 9, 4, 30, 35, 40, 45, 50],
              [54, 49, 44, 39, 34, 60, 65, 70, 75, 80], [84, 79, 74, 69, 64, 90, 95, 100, 105, 110],
              [2, 125, 128, 121, 62], [32, 123, 128, 127, 92]])
    return posmap


def make_pos2any():
    def add_seqs(graph, seqs):
        for seq in seqs:
            n = len(seq)
            for i, a in enumerate(seq):
                if i > 0:
                    graph[a].add(seq[i - 1])
                if i < n - 1:
                    graph[a].add(seq[i + 1])

    posmap = defaultdict(set)
    add_seqs(posmap,
             [[0, 1, 2, 3, 4], [20, 21, 22, 23, 24], [30, 31, 32, 33, 34], [50, 51, 52, 53, 54], [60, 61, 62, 63, 64],
              [80, 81, 82, 83, 84], [90, 91, 92, 93, 94], [110, 111, 112, 113, 114],
              [20, 15, 10, 5, 0, 126, 127, 120, 64, 69, 74, 79, 84],
              [24, 19, 14, 9, 4, 124, 123, 122, 60, 65, 70, 75, 80],
              [50, 45, 40, 35, 30, 124, 125, 126, 94, 99, 104, 109, 114],
              [54, 49, 44, 39, 34, 122, 121, 120, 90, 95, 100, 105, 110]
                 , [20, 15, 10, 5, 0, 94, 99, 104, 109, 114], [24, 19, 14, 9, 4, 30, 35, 40, 45, 50],
              [54, 49, 44, 39, 34, 60, 65, 70, 75, 80], [84, 79, 74, 69, 64, 90, 95, 100, 105, 110],
              [2, 125, 128, 121, 62], [32, 123, 128, 127, 92]])
    return posmap

def add_src(graph, seqs):
    for seq in seqs:
        graph[seq[0]].update(seq[1:])

def add_from_src(graph, seqs):
    for seq in seqs:
        for i in seq[1:]:
            graph[i].add(seq[0])

def make_pos2plain():
    def add_seqs(graph, seqs):
        for a, b in seqs:
            graph[a].add(b)
            graph[b].add(a)

    posmap = defaultdict(set)
    add_seqs(posmap, [[2, 7], [10, 11], [13, 14], [17, 22], [20, 25], [22, 27], [24, 29],
                      [32, 37], [40, 41], [43, 44], [47, 52], [50, 55], [52, 57], [54, 59],
                      [62, 67], [70, 71], [73, 74], [77, 82], [80, 85], [82, 87], [84, 89],
                      [92, 97], [100, 101], [103, 104], [107, 112], [110, 115], [112, 117], [114, 119]])
    add_src(posmap, [[6, 0, 1, 2, 5, 7, 10, 11], [8, 2, 3, 4, 7, 9, 13, 14], [12, 7, 11, 13, 17],
                     [16, 10, 11, 15, 17, 20, 21, 22], [18, 13, 14, 17, 19, 22, 23, 24],
                     [36, 30, 31, 32, 35, 37, 40, 41], [38, 32, 33, 34, 37, 39, 43, 44], [42, 37, 41, 43, 47],
                     [46, 40, 41, 45, 47, 50, 51, 52], [48, 43, 44, 47, 49, 52, 53, 54],
                     [66, 60, 61, 62, 65, 67, 70, 71], [68, 62, 63, 64, 67, 69, 73, 74], [72, 67, 71, 73, 77],
                     [76, 70, 71, 75, 77, 80, 81, 82], [78, 73, 74, 77, 79, 82, 83, 84],
                     [96, 90, 91, 92, 95, 97, 100, 101], [98, 92, 93, 94, 97, 99, 103, 104], [102, 97, 101, 103, 107],
                     [106, 100, 101, 105, 107, 110, 111, 112], [108, 103, 104, 107, 109, 112, 113, 114]])
    return posmap

def make_pos2camp():
    posmap = defaultdict(set)
    add_from_src(posmap,
                 [[6, 0, 1, 2, 5, 7, 10, 11, 12], [8, 2, 3, 4, 7, 9, 12, 13, 14], [12, 6, 7, 8, 11, 13, 16, 17, 18],
                  [16, 10, 11, 12, 15, 17, 20, 21, 22], [18, 12, 13, 14, 17, 19, 22, 23, 24],
                  [36, 30, 31, 32, 35, 37, 40, 41, 42], [38, 32, 33, 34, 37, 39, 42, 43, 44],
                  [42, 36, 37, 38, 41, 43, 46, 47, 48],
                  [46, 40, 41, 42, 45, 47, 50, 51, 52], [48, 42, 43, 44, 47, 49, 52, 53, 54],
                  [66, 60, 61, 62, 65, 67, 70, 71, 72], [68, 62, 63, 64, 67, 69, 72, 73, 74],
                  [72, 66, 67, 68, 71, 73, 76, 77, 78],
                  [76, 70, 71, 72, 75, 77, 80, 81, 82], [78, 72, 73, 74, 77, 79, 82, 83, 84],
                  [96, 90, 91, 92, 95, 97, 100, 101, 102], [98, 92, 93, 94, 97, 99, 102, 103, 104],
                  [102, 96, 97, 98, 101, 103, 106, 107, 108],
                  [106, 100, 101, 102, 105, 107, 110, 111, 112], [108, 102, 103, 104, 107, 109, 112, 113, 114]])
    return posmap


def make_pos2flag():
    posmap = defaultdict(set)
    add_from_src(posmap,
                 [[26, 21, 25, 27], [28, 23, 27, 29], [56, 51, 55, 57], [58, 53, 57, 59], [86, 81, 85, 87],
                  [88, 83, 87, 89],
                  [116, 111, 115, 117], [118, 113, 117, 119]])
    return posmap

def make_flag2pos():
    posmap = defaultdict(set)
    add_src(posmap,
                 [[26, 21, 25, 27], [28, 23, 27, 29], [56, 51, 55, 57], [58, 53, 57, 59], [86, 81, 85, 87],
                  [88, 83, 87, 89],
                  [116, 111, 115, 117], [118, 113, 117, 119]])
    return posmap

def make_action_mask():
    posmap = defaultdict(set)
    add_from_src(posmap, [[6, 0, 1, 2, 5, 6, 8, 9], [8, 2, 3, 4, 6, 7, 10, 11], [12, 6, 9, 10, 13],
                          [16, 8, 9, 12, 13, 15, 16, 17], [18, 10, 11, 13, 14, 17, 18, 19],
                          [66, 23, 24, 25, 28, 29, 31, 32], [68, 25, 26, 27, 29, 30, 33, 34], [72, 29, 32, 33, 36],
                          [76, 31, 32, 35, 36, 38, 39, 40], [78, 33, 34, 36, 37, 40, 41, 42]
                          ])
    posmap[0].update((126, 127, 120, 94))
    posmap[2].update((125, 128, 121))
    posmap[4].update((124, 123, 122, 30))
    posmap[23].update((124, 123, 122, 34))
    posmap[25].update((125, 128, 121))
    posmap[27].update((126, 127, 120, 90))

    action_mask = np.zeros(46 * 129, dtype=bool)
    for a, neighbors in posmap.items():
        base = a * 129
        action_mask[base + np.fromiter(neighbors, dtype=np.int32)] = True
    return action_mask


POS2STRAIGHT = make_pos2straight()
POS2ANY = make_pos2any()
POS2PLAIN = make_pos2plain()
POS2CAMP = make_pos2camp()
POS2FLAG = make_pos2flag()
FLAG2POS = make_flag2pos()
ACTION_MASK = make_action_mask()


def traverse_bfs(start_pos, pos2piece, graph):
    piece = pos2piece[start_pos]

    queue = deque([start_pos])
    visited = {start_pos}
    positions = set()
    pieces = set()
    parent = {}
    while queue:
        nxt = queue.popleft()
        for pos in graph[nxt]:
            if pos in visited:
                continue
            visited.add(pos)

            if pos not in pos2piece:
                queue.append(pos)
                positions.add(pos)
                parent[pos] = nxt
                continue
            piece1 = pos2piece[pos]
            pieces.add(piece1)
            if abs(piece[0] - piece1[0]) in {1, 3}:
                positions.add(pos)
                parent[pos] = nxt

    return positions, pieces, parent


def traverse_dfs(start_pos, pos2piece, graph):
    piece = pos2piece.get(start_pos)

    positions = set()
    pieces = set()
    parent = {}

    def dfs(tree, prev):
        for path, subtree in tree.items():
            cur_prev = prev
            for pos in path:
                if pos not in pos2piece:
                    positions.add(pos)
                    parent[pos] = cur_prev
                    cur_prev = pos
                    continue
                piece1 = pos2piece[pos]
                pieces.add(piece1)
                if piece and abs(piece[0] - piece1[0]) in {1, 3}:
                    positions.add(pos)
                    parent[pos] = cur_prev
                break
            else:
                dfs(subtree, cur_prev)

    dfs(graph[start_pos], start_pos)
    return positions, pieces, parent


def traverse(start_pos, pos2piece, graph, is_camp=False):
    piece = pos2piece.get(start_pos)

    positions = set()
    pieces = set()
    parent = {}
    for pos in graph[start_pos]:
        if pos not in pos2piece:
            positions.add(pos)
            parent[pos] = start_pos
            continue
        piece1 = pos2piece[pos]
        pieces.add(piece1)
        if not is_camp and piece and abs(piece[0] - piece1[0]) in {1, 3}:
            positions.add(pos)
            parent[pos] = start_pos

    return positions, pieces, parent


def get_moves(start_pos, pos2piece):
    positions1, pieces1, parent1 = traverse_dfs(start_pos, pos2piece, POS2STRAIGHT)
    positions2, pieces2, parent2 = traverse(start_pos, pos2piece, POS2PLAIN)
    positions3, pieces3, parent3 = traverse(start_pos, pos2piece, POS2CAMP, True)
    positions4, _, parent4 = traverse(start_pos, pos2piece, POS2FLAG)
    _, pieces4, _ = traverse(start_pos, pos2piece, FLAG2POS)

    positions = positions1 | positions2 | positions3 | positions4
    pieces = pieces1 | pieces2 | pieces3 | pieces4
    parent = {start_pos: None}
    for d in (parent1, parent2, parent3, parent4):
        for k, v in d.items():
            assert k not in parent, f"Duplicate key detected: {k}"
            parent[k] = v

    return positions, pieces, parent


def get_engineer_moves(start_pos, pos2piece):
    positions1, pieces1, parent1 = traverse_bfs(start_pos, pos2piece, POS2ANY)
    positions2, pieces2, parent2 = traverse(start_pos, pos2piece, POS2PLAIN)
    positions3, pieces3, parent3 = traverse(start_pos, pos2piece, POS2CAMP, True)
    positions4, _, parent4 = traverse(start_pos, pos2piece, POS2FLAG)
    _, pieces4, _ = traverse(start_pos, pos2piece, FLAG2POS)

    positions = positions1 | positions2 | positions3 | positions4
    pieces = pieces1 | pieces2 | pieces3 | pieces4
    parent = {start_pos: None}
    for d in (parent1, parent2, parent3, parent4):
        for k, v in d.items():
            assert k not in parent, f"Duplicate key detected: {k}"
            parent[k] = v

    return positions, pieces, parent


def transform_pos(pos):
    mask1 = pos < 120
    mask2 = (pos >= 120) & (pos <= 125)
    mask3 = (pos >= 126) & (pos <= 127)

    pos[mask1] = (pos[mask1] - 30) % 120
    pos[mask2] = pos[mask2] + 2
    pos[mask3] = pos[mask3] - 6

    return pos


def transform_pos_reverse(pos):
    mask1 = pos < 120
    mask2 = (pos >= 120) & (pos <= 121)
    mask3 = (pos >= 122) & (pos <= 127)

    pos[mask1] = (pos[mask1] + 30) % 120
    pos[mask2] = pos[mask2] + 6
    pos[mask3] = pos[mask3] - 2

    return pos


def transform_pos_reverse_scalar(pos):
    if pos < 120:
        return (pos + 30) % 120
    if 120 <= pos <= 121:
        return pos + 6
    if 122 <= pos <= 127:
        return pos - 2
    return pos


# role: 0雷 1司 2军 3师 4旅 5团 6营 7连 8排 9兵 10炸 11旗
def generate_layout():
    layout = [None] * 25
    roles = [0, 0, 0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10]

    def pop_with_range(roles, rule):
        candidates = [i for i, v in enumerate(roles) if rule(v)]
        if not candidates:
            return None

        idx = random.choice(candidates)
        return roles.pop(idx)

    flag_pos = random.choice([21, 23])
    layout[flag_pos] = 11
    other = 23 if flag_pos == 21 else 21

    role = pop_with_range(roles, lambda i: i == 0 or i == 7 or i == 8)
    layout[other] = role

    num = roles.count(0)
    roles = roles[num:]
    remove_num = random.sample([i for i in range(15, 25) if layout[i] == None], num)
    for i in remove_num:
        layout[i] = 0

    for pos in [i for i in range(15, 25) if layout[i] == None]:
        layout[pos] = pop_with_range(roles, lambda i: i >= 5)

    for pos in range(5):
        layout[pos] = pop_with_range(roles, lambda i: i != 10)

    for pos in range(5, 15):
        layout[pos] = pop_with_range(roles, lambda i: True)

    return layout

reward_map = {
    0:1.0, 1:15.0, 2:9.0, 3:5.0, 4:3.0, 5:2.0, 6:1.5, 7:1.2, 8:1.0, 9:2.0, 10:3.0, 11:0.0
}