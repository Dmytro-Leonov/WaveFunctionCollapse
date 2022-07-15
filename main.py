import os
import random
from PIL import Image
import numpy as np
from enum import Enum

TILES_FOLDER = "TileSets"


class Dir(Enum):
    TOP = 1
    RIGHT = 2
    BOTTOM = 3
    LEFT = 4


class Sides(str, Enum):
    TOP = "top"
    RIGHT = "right"
    BOTTOM = "bottom"
    LEFT = "left"


def chose_tile_set():
    all_tile_sets = os.listdir(TILES_FOLDER)
    if not all_tile_sets:
        raise OSError("No tile sets to choose")
    print("Choose tile set:")
    for i, tile_set in enumerate(all_tile_sets):
        print(f"{i + 1}: {tile_set}")
    while True:
        try:
            print("Enter choice: ", end="")
            choice = int(input())
            tile_set = all_tile_sets[choice - 1]
            break
        except ValueError:
            print("Invalid input")
        except IndexError:
            print("Invalid choice")

    all_tiles = np.array(Image.open(os.path.join(TILES_FOLDER, tile_set)))
    return all_tiles


def make_choice(question: str, answers: list) -> str:
    element = None
    print(question)
    for i, answer in enumerate(answers):
        print(f"{i + 1}: {answer}")
    while True:
        try:
            print("Enter choice: ", end="")
            choice = int(input())
            element = answers[choice - 1]
            break
        except ValueError:
            print("Invalid input")
        except IndexError:
            print("Invalid choice")
    return element


def pars_all_possible_tiles(tile_set, tile_size_px):
    # tiles = []
    # for tile_offset in range(tile_size_px, len(tile_set[0]), tile_size_px):
    #     tile = np.vstack([tile_set[:, tile_offset - tile_size_px:tile_offset]])
    #     tiles.append(tile)
    return


def pars_strict_tiles(tile_set, tile_size_px):
    tiles = []
    for tile_offset in range(tile_size_px, tile_set.shape[1] + 1, tile_size_px):
        tile = np.vstack([tile_set[:, tile_offset - tile_size_px:tile_offset]])
        tiles.append(tile)
    return tiles


class Tile:
    sides_dict = {}
    counter = 0

    def __init__(self, tile):
        self.tile = tile
        self.top = self.get_side(Dir.TOP)
        self.right = self.get_side(Dir.RIGHT)
        self.bottom = self.get_side(Dir.BOTTOM)
        self.left = self.get_side(Dir.LEFT)
        self.weight = 1
        self.possible = True

    def get_side(self, direction) -> int:
        side_hash = None
        match direction:
            case Dir.TOP:
                side_hash = "".join([str(channel).zfill(3) for pixel in self.tile[0] for channel in pixel])
            case Dir.BOTTOM:
                side_hash = "".join(
                    [str(channel).zfill(3) for pixel in self.tile[self.tile.shape[0] - 1] for channel in pixel]
                )
            case Dir.LEFT:
                side_hash = "".join([str(channel).zfill(3) for pixel in self.tile[:, 0] for channel in pixel])
            case Dir.RIGHT:
                side_hash = "".join(
                    [str(channel).zfill(3) for pixel in self.tile[:, self.tile.shape[1] - 1] for channel in pixel]
                )
        if side_hash in Tile.sides_dict:
            return Tile.sides_dict[side_hash]
        else:
            Tile.counter += 1
            Tile.sides_dict[side_hash] = Tile.counter
            return Tile.counter


class TileSet:
    def __init__(self, ):
        self.tiles = []

    def add(self, tile):
        for tile_in_set in self.tiles:
            if np.array_equal(tile_in_set.tile, tile.tile):
                tile_in_set.weight += 1
                return
        self.tiles.append(tile)

    def save(self):
        for i, tile in enumerate(self.tiles):
            Image.fromarray(tile.tile).save(f"tile{i}.png")

    def __str__(self):
        return "{}".format('\n'.join([str(tile.tile) for tile in self.tiles]))


def all_possible_tiles(tiles, rotate: bool, mirror: bool):
    tile_set = TileSet()
    if not rotate and not mirror:  # nothing
        for tile in tiles:
            tile_set.add(Tile(tile))
    elif rotate and mirror:  # 4 turns + 4 mirrors
        for tile in tiles:
            for _ in range(4):
                tile_set.add(Tile(tile))
                tile = np.rot90(tile, 1, (1, 0))
            tile = np.fliplr(tile)
            for _ in range(4):
                tile_set.add(Tile(tile))
                tile = np.rot90(tile, 1, (1, 0))
    elif rotate:  # 4 turns
        for tile in tiles:
            for _ in range(4):
                tile_set.add(Tile(tile))
                tile = np.rot90(tile, 1, (1, 0))
    else:  # only mirror
        for tile in tiles:
            tile_set.add(Tile(tile))
            tile_set.add(Tile(np.fliplr(tile)))
            tile_set.add(Tile(np.flipud(tile)))
    return tile_set


class Grid:
    def __init__(self, tile_set: TileSet, use_weights: bool, width: int, height: int):
        self.tile_set = tile_set
        self.grid = [[i for i in range(len(tile_set.tiles))] for _ in range(width * height)]
        self.use_weights = use_weights
        self.width = width
        self.height = height
        self.stack = []

    # find if wfc is collapsed
    def is_collapsed(self):
        for cell in self.grid:
            # check if any grid cell has more than one possible tile
            if len(cell) > 1:
                return False
        return True

    # choose random position out of cells with min entropy
    def min_entropy_pos(self):
        # list of possible positions
        pos_list = []
        # current min entropy
        min_entropy = len(self.tile_set.tiles)
        for i, cell in enumerate(self.grid):
            entropy = len(cell)
            # if there is new min entropy
            if min_entropy > entropy > 1:
                pos_list.clear()
                pos_list.append(i)
                min_entropy = entropy
            # if cell has the same entropy as current min entropy is
            elif entropy == min_entropy:
                pos_list.append(i)
        # return random pos of cell with min entropy
        return random.choice(pos_list)

    # collapse at given pos
    def collapse_at(self, pos: int):
        if self.use_weights:
            # if weights are used, calculate their sum to later normalize them
            total_weights = sum([self.tile_set.tiles[tile_pos].weight for tile_pos in self.grid[pos]])
            # choose random tile based on weights
            chosen_tile = np.random.choice(
                a=self.grid[pos],
                size=1,
                p=[self.tile_set.tiles[tile_pos].weight / total_weights for tile_pos in self.grid[pos]]
            )[0]  # take 1st element because function returns array instead of single element
        else:
            # just choose random tile
            chosen_tile = random.choice(self.grid[pos])
        # remove all tiles except of the chosen one
        self.grid[pos] = [chosen_tile]

    def valid_neighbours(self, pos: int) -> list[tuple[Dir, int]]:
        positions = []
        # if it is not on top
        if pos - self.width >= 0:
            # append position to the top
            positions.append((Dir.TOP, pos - self.width))
        # if it is not on far right
        if pos % self.width != self.width - 1:
            # append position to the right
            positions.append((Dir.RIGHT, pos + 1))
        # if it is not on bottom
        if pos + self.width < self.width * self.height:
            # append position to the bottom
            positions.append((Dir.BOTTOM, pos + self.width))
        # if it is not on far left
        if pos % self.width != 0:
            # append position to the left
            positions.append((Dir.LEFT, pos - 1))
        return positions

    def constraint(self, cur_pos: int, direction: Dir, neighbour_pos: int) -> bool:
        cur_side = ""
        neighbour_side = ""
        side = None
        # create lists with available sides of cell at current position and it's neighbour in relation to one another
        match direction:
            case Dir.TOP:
                cur_side = Sides.TOP
                neighbour_side = Sides.BOTTOM
            case Dir.RIGHT:
                cur_side = Sides.RIGHT
                neighbour_side = Sides.LEFT
            case Dir.BOTTOM:
                cur_side = Sides.BOTTOM
                neighbour_side = Sides.TOP
            case Dir.LEFT:
                cur_side = Sides.LEFT
                neighbour_side = Sides.RIGHT

        cur_pos_sides = [getattr(self.tile_set.tiles[tile_pos], cur_side) for tile_pos in self.grid[cur_pos]]
        neighbour_pos_sides = [getattr(self.tile_set.tiles[tile_pos], neighbour_side) for tile_pos in self.grid[neighbour_pos]]
        matched_sides = set(cur_side for cur_side in cur_pos_sides if cur_side in neighbour_pos_sides)
        # print(f"{cur_pos_sides=}\n{neighbour_pos_sides=}\n{matched_sides=}\n{self.grid[neighbour_pos]=}")
        new_neighbour_tiles_pos = []

        for tile_pos in self.grid[neighbour_pos]:
            if getattr(self.tile_set.tiles[tile_pos], neighbour_side) in matched_sides:
                new_neighbour_tiles_pos.append(tile_pos)
        # print(f"{new_neighbour_tiles_pos=}", end="\n\n")
        if self.grid[neighbour_pos] != new_neighbour_tiles_pos:
            self.grid[neighbour_pos] = new_neighbour_tiles_pos
            return True
        return False

    def propagate(self, pos: int) -> None:
        # append position to stack
        self.stack.append(pos)
        # while stack is not empty
        while self.stack:
            # print(self.stack)
            # pop position from stack
            cur_pos = self.stack.pop()
            # for direction and position in every valid neighbour
            # print(self.valid_neighbours(cur_pos))
            for d, neighbour_pos in self.valid_neighbours(cur_pos):
                # constraint neighbour
                changed = self.constraint(cur_pos, d, neighbour_pos)
                # append it's position to stack if it's not already there
                if neighbour_pos not in self.stack and changed:
                    self.stack.append(neighbour_pos)

    def iterate(self) -> None:
        pos = self.min_entropy_pos()
        self.collapse_at(pos)
        self.propagate(pos)

    def run(self):
        while not self.is_collapsed():
            self.iterate()

    def graphics(self):
        image = []
        for i in range(self.height):
            tiles_row = []
            for j in range(self.width):
                tiles_row.append(self.tile_set.tiles[self.grid[i * self.width + j][0]].tile)  # need to check here
            image.append(np.concatenate(tiles_row, axis=1))
        image = np.concatenate(image, axis=0)
        Image.fromarray(image).save(f"result.png")


def main():
    all_tiles = chose_tile_set()
    operation_choice = make_choice("Как прожарим мяско Сережи:", ["Жесткая разбивка", "Не жесткая разбивка"])
    if operation_choice == "Жесткая разбивка":
        tiles = pars_all_possible_tiles(all_tiles, 3)
    else:
        tiles = pars_strict_tiles(all_tiles, 8)
    tile_set = all_possible_tiles(tiles, rotate=False, mirror=True)
    print(Tile.sides_dict)
    wfc = Grid(tile_set, True, 20, 20)
    wfc.run()
    wfc.graphics()


if __name__ == '__main__':
    main()
