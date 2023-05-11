from collections import deque
from enum import IntEnum
from typing import Tuple, Iterable, Optional, Union, List

import gym
import numpy as np
from colorama import Fore, Back
from gym.core import RenderFrame, ObsType, ActType
from gym.spaces import Box
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from csidrl.agents.agents import RandomAgent


class HydraulicRotations(gym.Env):
    def __init__(self, size=64, checkpoints=None, solved=False, grid=None):
        self.size = size
        self.checkpoints = checkpoints
        self.water = np.zeros((size, size))

        if grid is None:
            self.grid = HydraulicRotationsEnv.generate(
                size, checkpoints=checkpoints, solved=solved
            )
        else:
            self.grid = grid
        self.last_action = None
        self.last_state = None

        flood_fill_water(self.grid, (-1, 0), self.water)

        self.observation_space = self.state_space

        self.max_water = 0

    @property
    def state_space(self):
        return gym.spaces.Box(
            low=0, high=1, shape=[self.size, self.size, 1 + len(TileKind)], dtype=int
        )

    @property
    def action_space(self):
        return gym.spaces.Discrete(n=(self.size * self.size))

    @staticmethod
    def generate(
        size=64, checkpoints=4, solved=False, grid=None, show_solved_also=False
    ):
        noise = get_random_state(size)

        path = list(get_random_path(size, checkpoints))

        path_printed = np.zeros((size, size), dtype=int)

        if show_solved_also:
            showcase_solved = np.zeros((size, size), dtype=int)

        for i in range(len(path)):
            x, y = path[i]
            prev_x, prev_y = path[i - 1] if (i > 0) else (-1, 0)
            next_x, next_y = path[i + 1] if (i < len(path) - 1) else (size - 1, size)

            p_delta = x - prev_x, y - prev_y
            n_delta = next_x - x, next_y - y

            if path_printed[x, y] != 0:
                path_printed[x, y] = TileKind.FourWay
                continue

            tile = get_pipe_by_in_out(p_delta, n_delta)

            if show_solved_also:
                showcase_solved[x, y] = tile.value

            if not solved:
                tile = shuffle_tile(TileKind(tile))

            path_printed[x, y] = tile.value

        noise[path_printed != 0] = 0

        if grid is None:
            grid = np.zeros((size, size), dtype=int)

        grid *= 0
        grid += noise
        grid += path_printed

        if show_solved_also:
            grid_solved = np.zeros((size, size), dtype=int)
            grid_solved += noise
            grid_solved += showcase_solved
            return grid, grid_solved

        return grid

    def reset(self, *args, **kwargs):
        self.max_water = 0
        self.water *= 0
        self.grid = HydraulicRotationsEnv.generate(
            self.size, checkpoints=self.checkpoints, solved=False, grid=self.grid
        )
        next_state, reward, terminated, trunc, info = self.step(
            np.zeros((self.size, self.size))
        )
        self.last_action = None
        self.last_state = None
        return next_state, info

    def step(self, action: np.ndarray):
        action = np.array(action)

        if action.reshape((-1,)).shape == (1,):
            action_arr = np.zeros((self.size * self.size), dtype=int)
            action_arr[action] = 1
            action = action_arr.reshape((self.size, self.size))
        else:
            action = action.reshape((self.size, self.size))
            action = action.astype(int)

        for x, y in np.array(action.nonzero()).T:
            rotate(self.grid, x, y)
            self.last_action = (x, y)

        flood_fill_water(self.grid, (-1, 0), self.water)

        done = self.water[self.size - 1, self.size - 1]

        grid = np.eye(len(TileKind) + 1)[self.grid]

        # Drop first channel, as that is useless and empty
        grid = grid[:, :, 1:]

        water = self.water.reshape((self.size, self.size, 1))
        next_state = np.concatenate((water, grid), axis=2)
        score = -1

        self.last_state = next_state

        done = bool(done)

        # water_level = np.sum(water)
        #
        # if water_level > self.max_water:
        #     score += water_level - self.max_water
        #     self.max_water = water_level
        #
        # if done:
        #     full_reward = self.size * self.size
        #     score += full_reward - self.max_water

        return next_state, score, done, done, {}

    def render(
        self, *args, **kwargs
    ) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        action = kwargs.get("last_action", self.last_action)
        return get_grid_repr_with_color(self.grid, self.water, action)

    @staticmethod
    def showcase(size=64, checkpoints=None):
        grid, grid_solved = HydraulicRotationsEnv.generate(
            size, checkpoints, show_solved_also=True
        )

        env = HydraulicRotationsEnv(size, checkpoints, grid=grid)
        env_solved = HydraulicRotationsEnv(size, checkpoints, grid=grid_solved)

        print(env.render())
        print(env_solved.render())


class RotationsBetterment(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.conversions = list(get_tile_conversions())
        self.size = env.observation_space.shape[:2]
        self.observation_space = Box(
            low=0,
            high=1,
            shape=(*self.size, 1 + COMPACT_BITS_KIND + COMPACT_BITS_DIR),
        )

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        obs, info = self.env.reset(**kwargs)
        return self.convert_observation(obs), info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        obs, reward, done, trunc, info = self.env.step(action)
        obs = self.convert_observation(obs)
        return obs, reward, done, trunc, info

    def convert_observation(self, observation) -> ObsType:
        water = observation[:, :, 0]
        full_repr = observation[:, :, 1:]
        compact_repr = np.zeros((*self.size, COMPACT_BITS_KIND + COMPACT_BITS_DIR))

        for full, compact in self.conversions:
            found = np.all(full_repr == full, axis=2)
            compact_repr[found, :] = compact

        return np.concatenate((water.reshape((*self.size, 1)), compact_repr), axis=2)


TileKind = IntEnum(
    "TileKind",
    [
        "Empty",
        "Straight_LeftRight",
        "Straight_UpDown",
        "Curved_UpLeft",
        "Curved_UpRight",
        "Curved_DownLeft",
        "Curved_DownRight",
        "T_UpLeftRight",
        "T_DownLeftRight",
        "T_UpDownLeft",
        "T_UpDownRight",
        "FourWay",
    ],
)

COMPACT_BITS_KIND = 5
COMPACT_BITS_DIR = 4


def get_tile_conversions():
    for index, tile_id in enumerate(TileKind):
        full = np.zeros((len(TileKind)))
        full[index] = 1
        compact = np.zeros((COMPACT_BITS_KIND + COMPACT_BITS_DIR))

        tile = TileKind(tile_id)

        if tile == "Empty":
            compact[0] = 1
        elif tile == "Straight_LeftRight":
            compact[1] = 1
            compact[COMPACT_BITS_KIND + 0] = 1
        elif tile == "Straight_UpDown":
            compact[1] = 1
            compact[COMPACT_BITS_KIND + 1] = 1

        elif tile == "Curved_UpLeft":
            compact[2] = 1
            compact[COMPACT_BITS_KIND + 0] = 1
        elif tile == "Curved_UpRight":
            compact[2] = 1
            compact[COMPACT_BITS_KIND + 1] = 1
        elif tile == "Curved_DownLeft":
            compact[2] = 1
            compact[COMPACT_BITS_KIND + 2] = 1
        elif tile == "Curved_DownRight":
            compact[2] = 1
            compact[COMPACT_BITS_KIND + 3] = 1

        elif tile == "T_UpLeftRight":
            compact[3] = 1
            compact[COMPACT_BITS_KIND + 0] = 1
        elif tile == "T_DownLeftRight":
            compact[3] = 1
            compact[COMPACT_BITS_KIND + 1] = 1
        elif tile == "T_UpDownLeft":
            compact[3] = 1
            compact[COMPACT_BITS_KIND + 2] = 1
        elif tile == "T_UpDownRight":
            compact[3] = 1
            compact[COMPACT_BITS_KIND + 3] = 1
        elif tile == "FourWay":
            compact[4] = 1

        yield full, compact


def get_pipe_repr(tile: TileKind):
    return {
        "Empty": " ",
        "Straight_LeftRight": "━",
        "Straight_UpDown": "┃",
        "Curved_UpLeft": "┛",
        "Curved_UpRight": "┗",
        "Curved_DownLeft": "┓",
        "Curved_DownRight": "┏",
        "T_UpLeftRight": "┻",
        "T_DownLeftRight": "┳",
        "T_UpDownLeft": "┫",
        "T_UpDownRight": "┣",
        "FourWay": "╋",
    }[tile.name]


def get_pipe_by_in_out(in_delta, out_delta, try_inverse=True):
    inx, iny = in_delta
    outx, outy = out_delta

    # Straights
    if iny == 0 and outy == 0:
        return TileKind.Straight_LeftRight
    if inx == 0 and outx == 0:
        return TileKind.Straight_UpDown

    # In Up ( Or out up when inversed )
    if iny == 1 and outx == 1:
        return TileKind.Curved_UpRight
    if iny == 1 and outx == -1:
        return TileKind.Curved_UpLeft

    # In Down
    if iny == -1 and outx == 1:
        return TileKind.Curved_DownRight
    if iny == -1 and outx == -1:
        return TileKind.Curved_DownLeft

    if try_inverse:
        inx *= -1
        iny *= -1
        outx *= -1
        outy *= -1
        return get_pipe_by_in_out((outx, outy), (inx, iny), try_inverse=False)

    raise Exception("Bug")


def get_grid_repr(grid):
    result = []

    for row in grid.T:
        for t in row:
            t_enum = TileKind(t.item())
            result.append(get_pipe_repr(t_enum))
        result.append("\n")

    return "".join(result)


def get_grid_repr_with_color(grid, water, last_action=None):
    result = []
    size_x, size_y = grid.shape

    for y in range(size_y):
        for x in range(size_x):
            t = TileKind(grid[x, y].item())

            tile_color = ""
            tile = get_pipe_repr(t)
            tile_reset = ""

            if water[x, y].item():
                tile_color = tile_color + Fore.BLUE
                tile_reset = tile_reset + Fore.RESET
            if last_action is not None and x == last_action[0] and y == last_action[1]:
                tile_color = tile_color + Back.CYAN
                tile_reset = tile_reset + Back.RESET

            result.append(f"{tile_color}{tile}{tile_reset}")
        result.append("\n")

    return "".join(result)


def get_grid_repr_with_rotation(grid, rotate_at):
    result = []
    size_x, size_y = grid.shape

    for y in range(size_y):
        for x in range(size_x):
            t = TileKind(grid[x, y].item())

            if x == rotate_at[0] and y == rotate_at[1]:
                result.append(Fore.RED + get_pipe_repr(t) + Fore.RESET)
                continue

            result.append(get_pipe_repr(t))

        result.append("\n")

    return "".join(result)


def shuffle_tile(tile: TileKind):
    group_straight = [TileKind.Straight_LeftRight, TileKind.Straight_UpDown]
    group_curve = [
        TileKind.Curved_UpLeft,
        TileKind.Curved_UpRight,
        TileKind.Curved_DownLeft,
        TileKind.Curved_DownRight,
    ]
    group_t = [
        TileKind.T_UpLeftRight,
        TileKind.T_DownLeftRight,
        TileKind.T_UpDownLeft,
        TileKind.T_UpDownRight,
    ]
    fw = [TileKind.FourWay]

    choices = {
        "Empty": [TileKind.Empty],
        "Straight_LeftRight": group_straight + group_t,
        "Straight_UpDown": group_straight + group_t,
        "Curved_UpLeft": group_curve + group_t,
        "Curved_UpRight": group_curve + group_t,
        "Curved_DownLeft": group_curve + group_t,
        "Curved_DownRight": group_curve + group_t,
        "T_UpLeftRight": group_t + fw,
        "T_DownLeftRight": group_t + fw,
        "T_UpDownLeft": group_t + fw,
        "T_UpDownRight": group_t + fw,
        "FourWay": fw,
    }

    valid = choices[tile.name]

    result = np.random.choice(valid)
    return TileKind(result)


def get_random_state(size) -> np.ndarray:
    kinds = len(TileKind)
    state = np.random.randint(low=1, high=kinds + 1, size=(size, size))
    return state


def get_random_path(size, checkpoints_n) -> Iterable[Tuple[int, int]]:
    checkpoints = np.random.randint(low=0, high=size, size=(checkpoints_n, 4))
    checkpoints = np.concatenate(
        (((0, 0, 0, 0),), checkpoints, ((size - 1, size - 1, size - 1, size - 1),)),
        axis=0,
    )

    for i in range(1, checkpoints.shape[0]):
        checkpoints[i][0] = checkpoints[i - 1][2]
        checkpoints[i][1] = checkpoints[i - 1][3]

    previous_yielded = (None, None)

    for px, py, x, y in checkpoints:
        grid = Grid(matrix=np.ones((size, size)))
        start = grid.node(px, py)
        end = grid.node(x, y)
        path, _ = AStarFinder().find_path(start, end, grid)

        for p in path:
            if not (p[0] == previous_yielded[0] and p[1] == previous_yielded[1]):
                yield p
                previous_yielded = p


def rotate(grid, x, y):
    tile = grid[x, y]

    group_straight = [TileKind.Straight_LeftRight, TileKind.Straight_UpDown]
    group_curve = [
        TileKind.Curved_UpLeft,
        TileKind.Curved_UpRight,
        TileKind.Curved_DownRight,
        TileKind.Curved_DownLeft,
    ]
    group_t = [
        TileKind.T_UpLeftRight,
        TileKind.T_UpDownRight,
        TileKind.T_DownLeftRight,
        TileKind.T_UpDownLeft,
    ]
    fw = [TileKind.FourWay]
    empty = [TileKind.Empty]

    tile = TileKind(tile)

    group = None
    for group_s in [group_straight, group_curve, group_t, fw, empty]:
        if tile in group_s:
            group = group_s
            break

    assert group is not None, f"Failed to rotate tile: {tile}"

    index = group.index(tile)

    rotated_tile = group[index + 1] if index + 1 < len(group) else group[0]
    grid[x, y] = rotated_tile.value


up = (0, -1)
down = (0, 1)
left = (-1, 0)
right = (1, 0)
water_dir_dict = {
    "Empty": [],
    "Straight_LeftRight": [left, right],
    "Straight_UpDown": [up, down],
    "Curved_UpLeft": [up, left],
    "Curved_UpRight": [up, right],
    "Curved_DownLeft": [down, left],
    "Curved_DownRight": [down, right],
    "T_UpLeftRight": [up, left, right],
    "T_DownLeftRight": [down, left, right],
    "T_UpDownLeft": [up, down, left],
    "T_UpDownRight": [up, down, right],
    "FourWay": [up, down, left, right],
}


def is_receptive(grid, x, y, from_x, from_y) -> bool:
    from_x, from_y = from_x * -1, from_y * -1

    water_dir = (from_x, from_y)
    tile = grid[x, y]
    tile_name = TileKind(tile).name
    return water_dir in water_dir_dict[tile_name]


def flood_fill_water(grid, starting_point, water):
    water *= 0

    water_tiles = deque([starting_point])

    while len(water_tiles) > 0:
        # print(len(water_tiles))
        x, y = water_tiles.pop()

        if (x, y) == starting_point:
            tile = TileKind.FourWay.name
        else:
            tile = TileKind(grid[x, y]).name

        water_deltas = water_dir_dict[tile]

        for next_delta_x, next_delta_y in water_deltas:
            next_x = next_delta_x + x
            next_y = next_delta_y + y

            if not (0 <= next_x < grid.shape[0]):
                continue

            if not (0 <= next_y < grid.shape[1]):
                continue

            if water[next_x, next_y].item() == 1:
                continue

            if not is_receptive(grid, next_x, next_y, next_delta_x, next_delta_y):
                continue

            if (next_x, next_y) in water_tiles:
                continue

            water[next_x, next_y] = 1
            water_tiles.append((next_x, next_y))

    return water

