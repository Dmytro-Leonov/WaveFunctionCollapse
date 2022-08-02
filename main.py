import os
import random
from PIL import Image
import cv2
import numpy as np
from natsort import os_sorted
from enum import Enum

# folders to choose tile sets from, store generated images, generated animations and frames
TILES_FOLDER = "TileSets"
GENERATED_IMAGES = "GeneratedImages"
GENERATED_ANIMATIONS = "GeneratedAnimations"
FRAMES = "Frames"


# enum for directions
class Dir(Enum):
    TOP = 1
    RIGHT = 2
    BOTTOM = 3
    LEFT = 4


# enum for tile sides
class Sides(str, Enum):
    TOP = "top"
    RIGHT = "right"
    BOTTOM = "bottom"
    LEFT = "left"


def chose_tile_set():
    all_tile_sets = os_sorted(os.listdir(TILES_FOLDER))
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

    all_tiles = np.array(Image.open(os.path.join(TILES_FOLDER, tile_set)).convert("RGB"))
    return all_tiles


def make_choice(question: str, answers: list) -> str:
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
    tiles = []
    for y_offset in range(tile_size_px, tile_set.shape[1] + 1, tile_size_px):
        for x_offset in range(tile_size_px, tile_set.shape[0] + 1, tile_size_px):
            tile = tile_set[y_offset - tile_size_px: y_offset, x_offset - tile_size_px: x_offset]
            tiles.append(tile)
    return tiles


def pars_strict_tiles(tile_set, tile_size_px):
    tiles = []
    for tile_offset in range(tile_size_px, tile_set.shape[1] + 1, tile_size_px):
        tile = tile_set[:, tile_offset - tile_size_px:tile_offset]
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
        self.avg_color = []
        self.weight = 1

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

    def add(self, tile: Tile):
        for tile_in_set in self.tiles:
            if np.array_equal(tile_in_set.tile, tile.tile):
                tile_in_set.weight += 1
                return
        tile.avg_color = tile.tile.mean(0).mean(0)
        self.tiles.append(tile)


def all_possible_tiles(tiles, rotate: bool, mirror: bool) -> TileSet:
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
    def __init__(self, tile_set: TileSet, use_weights: bool, tile_size: int, width: int, height: int, animate: bool):
        self.tile_set = tile_set
        self.grid = [[i for i in range(len(tile_set.tiles))] for _ in range(width * height)]
        self.use_weights = use_weights
        self.tile_size = (tile_size, tile_size)
        self.width = width
        self.height = height
        self.animate = animate
        self.stack = []
        self.frame = 0

        if self.animate:
            self.clear_frames()

    @staticmethod
    def clear_frames():
        for frame in os.listdir(FRAMES):
            os.remove(os.path.join(FRAMES, frame))

    # choose random position out of cells with min entropy
    def min_entropy_pos(self) -> int:
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
            # if cell has the same entropy as current min entropy
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
        neighbour_pos_sides = [getattr(self.tile_set.tiles[tile_pos], neighbour_side) for tile_pos in
                               self.grid[neighbour_pos]]
        matched_sides = set(cur_side for cur_side in cur_pos_sides if cur_side in neighbour_pos_sides)

        new_neighbour_tiles_pos = []

        for tile_pos in self.grid[neighbour_pos]:
            if getattr(self.tile_set.tiles[tile_pos], neighbour_side) in matched_sides:
                new_neighbour_tiles_pos.append(tile_pos)

        if self.grid[neighbour_pos] != new_neighbour_tiles_pos:
            self.grid[neighbour_pos] = new_neighbour_tiles_pos
            return True
        return False

    def propagate(self, pos: int) -> None:
        # append position to stack
        self.stack.append(pos)
        # while stack is not empty
        while self.stack:
            # pop position from stack
            cur_pos = self.stack.pop()
            # for direction and position in every valid neighbour
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

    # find max integer in in all files names in a folder, ex: 0.png, 1.png, 2.png -> 2
    @staticmethod
    def chose_max_index(folder: str, lstrip: str) -> int:
        max_index = 0
        # list all files in a folder
        for im in os.listdir(folder):
            try:
                # convert number in file name to integer removing part before and after it
                current = int(im.lstrip(lstrip).split(".")[0])
                if current > max_index:
                    max_index = current
            except ValueError:
                pass
        return max_index

    def make_animation(self) -> None:
        max_index = self.chose_max_index(GENERATED_ANIMATIONS, "animation_")
        # set fourcc for video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # make output file name
        filename = f"{GENERATED_ANIMATIONS}/animation_{max_index + 1}.mp4"
        # set up animation
        animation = cv2.VideoWriter(
            filename=filename,
            apiPreference=0,
            fourcc=fourcc,
            fps=100,
            frameSize=(self.tile_size[0] * self.width, self.tile_size[1] * self.height)
        )
        # add every frame to the animation
        for frame in os_sorted(os.listdir(FRAMES)):
            animation.write(np.array(cv2.imread(os.path.join(FRAMES, frame))))
        animation.release()
        print(f"Animation is saved as {filename}")

    def progress_bar(self, title, bar_length=20):
        total = self.width * self.height
        current = 0
        for cell in self.grid:
            if len(cell) == 1:
                current += 1
        fraction = current / total
        progress = int(fraction * bar_length) * "█"
        padding = int(bar_length - len(progress)) * "░"
        ending = "\n" if current == total else "\r"
        print(f"{title} [{progress}{padding}] {fraction * 100:.2f}%", end=ending)

    # find if wfc is collapsed and there are no cells with zero entropy
    def is_collapsed(self) -> (bool, bool):
        is_collapsed = True
        is_valid = True
        for cell in self.grid:
            # calculate number of possible tiles for the cell
            possible_tiles = len(cell)
            # if any grid cell has more than one or zero possible tiles
            if possible_tiles > 1 and is_collapsed:
                is_collapsed = False
            # if there are no possible tiles for cell
            elif not possible_tiles and is_valid:
                is_valid = False
        return is_collapsed, is_valid

    def is_valid(self) -> bool:
        # iterate over the whole grid
        for i in range(self.height):
            for j in range(self.width):
                # if there are no possible tiles for cell
                if not self.grid[i * self.width + j]:
                    return False
        return True

    def run(self) -> bool:
        while True:
            is_collapsed, is_valid = self.is_collapsed()

            if not is_valid:
                return False
            if is_collapsed:
                break

            self.iterate()

            if self.animate:
                self.progress_bar("Generating frames:")
                self.graphics(final=False)
            else:
                self.progress_bar("Generating image:")

        self.graphics()
        if self.animate:
            self.make_animation()
        return True

    # calculate average color for a given cell
    def calculate_avg_color(self, cell: list[int]) -> tuple[np.array]:
        avg_colors = []
        for tile_pos in cell:
            # calculate average color for every tile in the cell
            avg_colors.append(self.tile_set.tiles[tile_pos].avg_color)
        # calculate average color among all average colors of tiles converting it to int
        return tuple(np.array(avg_colors).mean(0).astype(int))

    def graphics(self, final=True) -> None:
        # variable to store image
        image = []
        # iterate over the whole grid
        for i in range(self.height):
            # variable to store row of tiles
            tiles_row = []
            for j in range(self.width):
                # get grid cell at given index
                cell = self.grid[i * self.width + j]
                # if there is only one possible tile that can be placed in the cell
                if len(cell) == 1:
                    # get tile from tile set and append it to the row
                    tiles_row.append(self.tile_set.tiles[self.grid[i * self.width + j][0]].tile)
                else:
                    # if there is more than one possible tiles for the cell
                    # generate tile of average color out of all possible tiles of cell and append it to the row
                    tiles_row.append(Image.new("RGB", self.tile_size, self.calculate_avg_color(cell)))
            # concatenate tiles vertically and append to image as a row of tiles
            image.append(np.concatenate(tiles_row, axis=1))
        # concatenate all rows in image horizontally
        image = np.concatenate(image, axis=0)

        # if this is not a final image (just a frame)
        if not final:
            # save it to the frames folder
            Image.fromarray(image).save(f"{FRAMES}/frame_{self.frame}.png")
            # increase frames counter
            self.frame += 1
        else:
            # if this is a final image (completely generated)
            max_index = self.chose_max_index(GENERATED_IMAGES, "result_")
            # save generated image with new max integer in name
            filename = f"{GENERATED_IMAGES}/result_{max_index + 1}.png"
            Image.fromarray(image).save(filename)
            print(f"Generated image is saved as {filename}")


def main():
    tile_size = 32
    width, height = (10, 10)
    animate = True
    all_tiles = chose_tile_set()
    operation_choice = make_choice("Как прожарим мяско Сережи:", ["Жесткая разбивка", "Не жесткая разбивка"])
    if operation_choice == "Жесткая разбивка":
        tiles = pars_all_possible_tiles(all_tiles, tile_size)
    else:
        tiles = pars_strict_tiles(all_tiles, tile_size)
    tile_set = all_possible_tiles(tiles, rotate=False, mirror=False)
    wfc = Grid(tile_set, True, tile_size, width, height, animate)
    if wfc.run():
        print("WFC finished successfully")
    else:
        print("\nCouldn't generate an image from a given tile set")

# before improvement 3.654643690001103
# after improvement 1.9025049199990463
if __name__ == '__main__':
    main()
