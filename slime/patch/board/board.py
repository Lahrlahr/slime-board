import numpy as np

from .board_util import *


# role: 0雷 1司 2军 3师 4旅 5团 6营 7连 8排 9兵 10炸 11旗
# color: 0橙 1紫 2绿 3蓝
# result：0移动 1吃 2被吃 3一起挂 4跳过 亮旗(0-8) 死亡(0-3)
# 掩码更新：最开始正确。两条路径都不会更新 地雷 和 不在掩码范围内的piece。 寻找路径除了地雷，都能正确,包括在flag中的piece
class Board(object):
    def __init__(self, layout=None):
        if layout is None:
            layout = [generate_layout() for _ in range(4)]
        self.layout = layout
        self.pos2piece = {}
        self.piece2pos = {}
        self.action2piece = [{}, {}]
        self.piece2action = [{}, {}]
        self.init_layout(layout)

        self.action_mask = [ACTION_MASK.copy() for _ in range(2)]
        self.init_action_mask()

        self.influenced_pieces = [set() for _ in range(4)]
        self.engineer_pieces = [set() for _ in range(4)]
        self.init_engineer_pieces()

        self.dead_players = set()
        self.step_count = [0, 0, 0, 0]
        self.pieces_count = [20, 20, 20, 20]
        self.flag_pos = [0, 0, 0, 0]
        self.init_pieces_count()

    def init_layout(self, layout):
        positions = [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 13, 14, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        counter = defaultdict(int)
        action_ids = [0, 0]
        for player_id, roles in enumerate(layout):
            base = 30 * player_id
            group = player_id % 2
            for i, (role, pos) in enumerate(zip(roles, positions)):
                idx = counter[(player_id, role)]
                counter[(player_id, role)] += 1
                piece = (player_id, role, idx)

                board_pos = base + pos
                self.piece2pos[piece] = board_pos
                self.pos2piece[board_pos] = piece

                if pos not in {26, 28}:
                    action_id = action_ids[group]
                    self.action2piece[group][action_id] = piece
                    self.piece2action[group][piece] = action_id
                    action_ids[group] += 1

    def init_action_mask(self):
        pos2piece = self.pos2piece
        action_mask = self.action_mask
        piece2action = self.piece2action

        engineer_positions = {
            0: [0, 2, 4, 60, 62, 64],
            1: [30, 32, 34, 90, 92, 94]
        }
        action_offsets = np.arange(120, 129)
        for group_id, positions in engineer_positions.items():
            mask = action_mask[group_id]
            for pos in positions:
                piece = pos2piece[pos]
                if piece[1] == 9:
                    indices = piece2action[group_id][piece] * 129 + action_offsets
                    mask[indices] = True

        mine_positions = {
            0: [20, 21, 22, 23, 24, 80, 81, 82, 83, 84],
            1: [50, 51, 52, 53, 54, 110, 111, 112, 113, 114]
        }
        action_offsets = np.arange(129)
        for group_id, positions in mine_positions.items():
            mask = action_mask[group_id]
            for pos in positions:
                piece = pos2piece[pos]
                if piece[1] == 0:
                    indices = piece2action[group_id][piece] * 129 + action_offsets
                    mask[indices] = False

    def init_engineer_pieces(self):
        for p in self.piece2pos:
            if p[1] == 9:
                self.engineer_pieces[p[0]].add(p)

    def init_pieces_count(self):
        pos2piece = self.pos2piece
        pieces_count = self.pieces_count
        flag_pos = self.flag_pos
        flag_positions = {
            0: [26, 28],
            1: [56, 58],
            2: [86, 88],
            3: [116, 118]
        }
        for player, positions in flag_positions.items():
            for i, pos in enumerate(positions):
                if pos2piece[pos][1] == 0:
                    pieces_count[player] += 1
                if pos2piece[pos][1] == 11:
                    flag_pos[player] = i + player * 2

    def get_layout(self, group):
        if group == 0:
            return self.layout[0] + self.layout[2]
        else:
            return self.layout[1] + self.layout[3]

    def update_influenced_pieces(self, pieces, positions):
        influenced_pieces = self.influenced_pieces
        pos2piece = self.pos2piece

        for piece in pieces:
            if piece[1] == 0:
                continue
            influenced_pieces[piece[0]].add(piece)

        for pos in positions:
            _, pieces, _ = get_moves(pos, pos2piece)
            for piece in pieces:
                if piece[1] == 0:
                    continue
                influenced_pieces[piece[0]].add(piece)

    def update_mask_by_pieces(self, pieces):
        pos2piece = self.pos2piece
        piece2pos = self.piece2pos
        piece2action = self.piece2action
        action_mask = self.action_mask

        for piece in pieces:
            if piece[1] == 0:
                continue

            group = piece[0] % 2
            action_id = piece2action[group].get(piece)
            if action_id is None:
                continue

            mask = action_mask[group]
            base = action_id * 129
            mask[base: base + 129] = False

            if piece not in piece2pos:
                continue

            pos = piece2pos[piece]
            if piece[1] == 9:
                positions, _, _ = get_engineer_moves(pos, pos2piece)
            else:
                positions, _, _ = get_moves(pos, pos2piece)

            positions = np.fromiter(positions, dtype=int)
            if group:
                transform_pos(positions)
            mask[base + positions] = True

    def update_mask(self, del_pieces):
        pieces = set(self.piece2pos.keys()).union(*self.influenced_pieces, del_pieces)
        self.update_mask_by_pieces(pieces)

        for influenced_pieces in self.influenced_pieces:
            influenced_pieces.clear()

    def update_dead_players(self, player):
        piece2pos = self.piece2pos
        pos2piece = self.pos2piece
        rm_pieces = [p for p in piece2pos if p[0] == player]
        reward = 0.0
        for rm_piece in rm_pieces:
            rm_pos = piece2pos[rm_piece]
            del piece2pos[rm_piece]
            del pos2piece[rm_pos]
            if rm_pos not in {26, 28, 56, 58, 86, 88, 116, 118}:
                reward += reward_map[rm_piece[1]]
        self.dead_players.add(player)
        self.influenced_pieces[player].update(rm_pieces)
        return reward

    def decode(self, player, action):
        group = player % 2
        action_id = action // 129
        pos = action % 129
        piece = self.action2piece[group][action_id]
        if group == 1:
            pos = transform_pos_reverse_scalar(pos)
        return piece, pos

    def get_path(self, piece, pos1):
        pos = self.piece2pos[piece]
        if piece[1] == 9:
            _, _, parent = get_engineer_moves(pos, self.pos2piece)
        else:
            _, _, parent = get_moves(pos, self.pos2piece)

        path = []
        cur = pos1
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)

        return path[::-1]

    def update(self, player, action):
        if action == 5934:
            self.step_count[player] += 1
            if self.step_count[player] == 5:
                reward = -self.update_dead_players(player)
                self.update_mask([])
                return 4, [], [], [player], reward
            return 4, [], [], [], -2.0

        piece, pos1 = self.decode(player, action)
        path = self.get_path(piece, pos1)
        if len(path) == 1:
            piece, pos1 = self.decode(player, action)
            path = self.get_path(piece, pos1)
        to_flag = pos1 in {26, 28, 56, 58, 86, 88, 116, 118}

        piece2pos = self.piece2pos
        pos2piece = self.pos2piece
        pos = piece2pos[piece]
        assert (pos != pos1)
        pieces_count = self.pieces_count
        reward = 0.0

        if pos1 not in pos2piece:
            piece2pos[piece] = pos1
            del pos2piece[pos]
            pos2piece[pos1] = piece

            if to_flag:
                reward -= reward_map[piece[1]]
                pieces_count[player] -= 1
                if pieces_count[player] == 0:
                    reward -= self.update_dead_players(player)
                    self.update_mask([])
                    return 0, path, [], [player], reward
            self.update_influenced_pieces([piece], [pos, pos1])
            return 0, path, [], [], reward

        piece1 = pos2piece[pos1]
        player1 = piece1[0]
        role = piece[1]
        role1 = piece1[1]
        dead_players = []
        flags = []
        if role1 == 11:
            if role == 10:
                del piece2pos[piece]
                del piece2pos[piece1]
                del pos2piece[pos]
                del pos2piece[pos1]
            else:
                piece2pos[piece] = pos1
                del piece2pos[piece1]
                del pos2piece[pos]
                pos2piece[pos1] = piece
            reward -= reward_map[role]

            self.pieces_count[player] -= 1
            if self.pieces_count[player] == 0:
                reward -= self.update_dead_players(player)
                dead_players.append(player)
            reward += self.update_dead_players(player1)
            dead_players.append(player1)
            self.update_mask([piece, piece1])
            if role == 10:
                return 3, path, [], dead_players, reward
            return 1, path, [], dead_players, reward
        elif role == 10 or role1 == 10 or role == role1:
            del piece2pos[piece]
            del piece2pos[piece1]
            del pos2piece[pos]
            del pos2piece[pos1]
            reward -= reward_map[role]
            if not to_flag:
                reward += reward_map[role1]

            pieces_count[player] -= 1
            if pieces_count[player] == 0:
                reward -= self.update_dead_players(player)
                dead_players.append(player)

            if role1 != 0 and not to_flag:
                pieces_count[player1] -= 1
                if pieces_count[player1] == 0:
                    reward += self.update_dead_players(player1)
                    dead_players.append(player1)

            if dead_players:
                self.update_mask([piece, piece1])
            else:
                self.update_influenced_pieces([piece, piece1], [pos, pos1])

            if role == 1 and player not in dead_players:
                flags.append(self.flag_pos[player])
            if role1 == 1 and player1 not in dead_players:
                flags.append(self.flag_pos[player1])
            return 3, path, flags, dead_players, reward
        elif (role == 9 and role1 == 0) or (role < role1 and not (role == 0 and role1 == 9)):
            piece2pos[piece] = pos1
            del piece2pos[piece1]
            del pos2piece[pos]
            pos2piece[pos1] = piece
            if not to_flag:
                reward += reward_map[role1]

            if to_flag:
                reward -= reward_map[role]
                pieces_count[player] -= 1
                if pieces_count[player] == 0:
                    reward -= self.update_dead_players(player)
                    dead_players.append(player)

            if role1 != 0 and not to_flag:
                pieces_count[player1] -= 1
                if pieces_count[player1] == 0:
                    reward += self.update_dead_players(player1)
                    dead_players.append(player1)

            if dead_players:
                self.update_mask([piece1])
            else:
                self.update_influenced_pieces([piece, piece1], [pos, pos1])
            return 1, path, [], dead_players, reward
        else:
            del piece2pos[piece]
            del pos2piece[pos]
            reward -= reward_map[role]

            pieces_count[player] -= 1
            if pieces_count[player] == 0:
                reward -= self.update_dead_players(player)
                dead_players.append(player)
                self.update_mask([piece])
            else:
                self.update_influenced_pieces([piece], [pos])

            if role == 1 and player not in dead_players:
                flags.append(self.flag_pos[player])
            return 2, path, flags, dead_players, reward

    def get_action_mask(self, player):
        if player in self.dead_players:
            return None

        pieces = self.influenced_pieces[player] | self.engineer_pieces[player]
        self.update_mask_by_pieces(pieces)

        self.influenced_pieces[player].clear()

        group = player % 2
        mask = self.action_mask[group]
        result = np.zeros(mask.size + 1, dtype=bool)
        half = mask.size // 2
        if player < 2:
            result[:half] = mask[:half]
        else:
            result[half:-1] = mask[half:]
        result[-1] = True
        return result


if __name__ == "__main__":
    board = Board()
    for i in range(10000):
        player = i % 4
        action_mask = board.get_action_mask(player)
        if action_mask is None:
            continue
        valid_actions = np.where(action_mask)[0]
        action = random.choice(valid_actions)
        board.update(player, action)
