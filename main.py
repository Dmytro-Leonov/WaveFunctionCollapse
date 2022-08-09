import datetime
import os
import time

from PIL import Image
import cv2
import numpy as np
import random
from natsort import os_sorted
from enum import Enum

# folders to choose tile sets from, store generated images, generated animations and frames
TILE_SETS_FOLDER = "TileSets"
GENERATED_IMAGES = "GeneratedImages"
GENERATED_ANIMATIONS = "GeneratedAnimations"
FRAMES = "Frames"

# settings
MIN_TILE_SIZE, MAX_TILE_SIZE = 2, 100
MIN_FIELD_WIDTH, MAX_FIELD_WIDTH = 2, 500
MIN_FIELD_HEIGHT, MAX_FIELD_HEIGHT = MIN_FIELD_WIDTH, MAX_FIELD_WIDTH
MIN_FPS, MAX_FPS = 1, 200
MIN_RETRIES, MAX_RETRIES = 1, 100


# enum for directions
class Dir(Enum):
    TOP = 1
    RIGHT = 2
    BOTTOM = 3
    LEFT = 4


def chose_tile_set() -> np.array:
    """
    Lists all tile sets in a folder and waits for user to choose one.
    If input or choice is invalid, user will see corresponding Error and will be prompted to enter new choice.
    If choice is valid, corresponding tile set opens with PIL, converts to RGB format and than to numpy array.
    Function returns opened tile set.
    """
    all_tile_sets = os_sorted(os.listdir(TILE_SETS_FOLDER))
    if not all_tile_sets:
        raise OSError("No tile sets to choose")
    print("Choose tile set:")
    for i, tile_set in enumerate(all_tile_sets):
        print(f"{i + 1}: {tile_set}")
    while True:
        try:
            print("Enter choice: ", end="")
            choice = int(input())
            tile_set_name = all_tile_sets[choice - 1]
            break
        except ValueError:
            print("Invalid input")
        except IndexError:
            print("Invalid choice")
    tile_set = np.array(Image.open(os.path.join(TILE_SETS_FOLDER, tile_set_name)).convert("RGB"))
    return tile_set


def pars_tiles(tile_set: np.array, tile_size_px: int) -> list[np.array]:
    """
    Parses tiles from a given tile set
    """
    # get height and width of tile set
    height, width = tile_set.shape[:2]
    # validate tile size (width and height should be divisible by it)
    if width % tile_size_px or height % tile_size_px:
        raise ValueError(
            f"Invalid tile size ({tile_size_px}) for tile set of given dimensions (width = {width}, height = {height})"
        )
    # list to store all parsed tiles
    tiles = []
    # iterate over tile set with offset of tile size
    for y_offset in range(tile_size_px, height + 1, tile_size_px):
        for x_offset in range(tile_size_px, width + 1, tile_size_px):
            # slice tile from tile set
            tile = tile_set[y_offset - tile_size_px: y_offset, x_offset - tile_size_px: x_offset]
            # append tile to tiles list
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

    def get_side(self, direction: Dir) -> int:
        """
        Hashes tile side using R, G and B values of every it's pixel.
        For example, if the side is [[23, 54, 23], [255, 255, 3], [0, 0, 0]],
        the hash would be "023054023255255003000000".
        Than it checks if this hash already exists in dictionary.
        If it isn't, than it gets added to the dictionary as a key with increased counter as a value that then returns.
        If it is, than function returns corresponding value.
        """
        side_hash = ""
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
    def __init__(self):
        self.tiles = []

    def add(self, tile: Tile):
        """
        Adds tile to the tile set
        """
        for tile_in_set in self.tiles:
            # if tile already exists in the tile set
            if np.array_equal(tile_in_set.tile, tile.tile):
                # increase it's weight
                tile_in_set.weight += 1
                return
        # calculate average color for tile to improve animation speed
        tile.avg_color = tile.tile.mean(0).mean(0)
        # add tile to the tile set
        self.tiles.append(tile)


def all_possible_tiles(tiles: list, rotate: bool, mirror: bool) -> TileSet:
    """
    Rotates and mirrors every tile depending on chosen modes.
    """
    tile_set = TileSet()
    if not rotate and not mirror:  # nothing
        # add every tile to tile set
        for tile in tiles:
            # add tile to tile set
            tile_set.add(Tile(tile))
    elif rotate and mirror:  # 4 turns and 4 mirrors
        for tile in tiles:
            for _ in range(4):
                # add tile to tile set
                tile_set.add(Tile(tile))
                # rotate clockwise
                tile = np.rot90(tile, 1, (1, 0))
            # mirror tile left to right
            tile = np.fliplr(tile)
            for _ in range(4):
                # add tile to tile set
                tile_set.add(Tile(tile))
                # rotate clockwise
                tile = np.rot90(tile, 1, (1, 0))
    elif rotate:  # 4 turns
        for tile in tiles:
            for _ in range(4):
                # add tile to tile set
                tile_set.add(Tile(tile))
                # rotate clockwise
                tile = np.rot90(tile, 1, (1, 0))
    else:  # only mirror
        for tile in tiles:
            # add tile to tile set
            tile_set.add(Tile(tile))
            # mirror tile left to right and add to tile set
            tile_set.add(Tile(np.fliplr(tile)))
            # mirror tile up to down and add to tile set
            tile_set.add(Tile(np.flipud(tile)))
    return tile_set


class WFC:
    def __init__(
            self, tile_set: TileSet, use_weights: bool, tile_size: int, width: int, height: int, animate: bool, fps: int
    ):
        self.tile_set = tile_set
        self.grid = [[i for i in range(len(tile_set.tiles))] for _ in range(width * height)]
        self.use_weights = use_weights
        self.tile_size = (tile_size, tile_size)
        self.width = width
        self.height = height
        self.animate = animate
        self.stack = []
        self.frame = 0
        self.fps = fps

    @staticmethod
    def clear_frames():
        """
        Clears folder with frames
        """
        for frame in os.listdir(FRAMES):
            os.remove(os.path.join(FRAMES, frame))

    def min_entropy_pos(self) -> int:
        """
        Returns random position out of cells with min entropy
        """
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

    def collapse_at(self, pos: int):
        """
        Takes cell position and leaves only one randomly chosen tile in it
        """
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
        """
        Takes position of cell and return list of tuples of direction and position of a valid neighbour cell.
        For example, if current position is '0', then valid neighbours are to the bottom and to the right.
        """
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
        """
        Removes positions of tiles that can't be matched with tiles in current cell from the neighbour cell
        in given direction.
        Returns True if the neighbour cell was changed and False otherwise.
        """
        new_neighbour_tiles_pos = []
        match direction:
            case Dir.TOP:
                # create list of sides in given direction for tiles in current cell
                cur_pos_sides = [self.tile_set.tiles[tile_pos].top for tile_pos in self.grid[cur_pos]]
                # iterate over positions of tiles in neighbour cell
                for tile_pos in self.grid[neighbour_pos]:
                    # if opposite side to given direction of tile matches side in created list of side
                    if self.tile_set.tiles[tile_pos].bottom in cur_pos_sides:
                        # than it can be placed there and it position in the tile set appends to the list
                        new_neighbour_tiles_pos.append(tile_pos)
            # same code for the other three sides
            case Dir.RIGHT:
                cur_pos_sides = [self.tile_set.tiles[tile_pos].right for tile_pos in self.grid[cur_pos]]
                for tile_pos in self.grid[neighbour_pos]:
                    if self.tile_set.tiles[tile_pos].left in cur_pos_sides:
                        new_neighbour_tiles_pos.append(tile_pos)
            case Dir.BOTTOM:
                cur_pos_sides = [self.tile_set.tiles[tile_pos].bottom for tile_pos in self.grid[cur_pos]]
                for tile_pos in self.grid[neighbour_pos]:
                    if self.tile_set.tiles[tile_pos].top in cur_pos_sides:
                        new_neighbour_tiles_pos.append(tile_pos)
            case Dir.LEFT:
                cur_pos_sides = [self.tile_set.tiles[tile_pos].left for tile_pos in self.grid[cur_pos]]
                for tile_pos in self.grid[neighbour_pos]:
                    if self.tile_set.tiles[tile_pos].right in cur_pos_sides:
                        new_neighbour_tiles_pos.append(tile_pos)

        # if there are changes to the neighbour cell
        if self.grid[neighbour_pos] != new_neighbour_tiles_pos:
            # replace tile positions with the new ones
            self.grid[neighbour_pos] = new_neighbour_tiles_pos
            return True
        # if there are no changes
        return False

    def propagate(self, pos: int) -> None:
        """
        Takes position of the cell that was collapsed and constraints neighbours until in every cell there are left
        only positions of tiles that can be matched together.
        """
        # append position to stack
        self.stack.append(pos)
        # while stack is not empty
        while self.stack:
            # pop position from stack
            cur_pos = self.stack.pop()
            # for direction and position in every valid neighbour
            for d, neighbour_pos in self.valid_neighbours(cur_pos):
                # constraint neighbour cell
                changed = self.constraint(cur_pos, d, neighbour_pos)
                # append it's position to stack if it's not already there
                # and the cell has changed in constraint() method
                if neighbour_pos not in self.stack and changed:
                    self.stack.append(neighbour_pos)

    def iterate(self) -> None:
        """
        One iteration of algorithm
        """
        pos = self.min_entropy_pos()
        self.collapse_at(pos)
        self.propagate(pos)

    @staticmethod
    def chose_max_index(folder: str, lstrip: str) -> int:
        """
        Takes folder name and part to strip from left side to leave only and integer.

        Finds max integer in in all files names in a given folder.
        For example, result_0.png, result_1.png, result_2.png -> 2
        """
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

    def generate_animation(self) -> None:
        """
        Generates animation from frames i folder using cv2
        """
        print("Generating animation...")
        # calculate max index in animations names in folder
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
            fps=self.fps,
            frameSize=(self.tile_size[0] * self.width, self.tile_size[1] * self.height)
        )
        # add every frame to the animation
        for frame in os_sorted(os.listdir(FRAMES)):
            animation.write(np.array(cv2.imread(os.path.join(FRAMES, frame))))
        animation.release()
        print(f"Animation is saved as {filename}")

    def progress_bar(self, title: str, current: int, bar_length=20) -> None:
        """
        Prints progress bar for a WFC
        """
        total = self.width * self.height
        fraction = current / total
        progress = int(fraction * bar_length) * "█"
        padding = int(bar_length - len(progress)) * "░"
        ending = "\n" if current == total else "\r"
        print(f"{title} [{progress}{padding}] {fraction * 100:.2f}%", end=ending)

    def is_collapsed(self) -> (bool, bool, int):
        """
        Calculates and returns in tuple:
            Whether WFC is collapsed (every cell has only one possible tile)

            Whether WFC is valid (there are no cells with zero entropy)

            How many cells are currently calculated
        """
        is_collapsed = True
        is_valid = True
        currently_done = 0
        for cell in self.grid:
            # calculate number of possible tiles for the cell
            possible_tiles = len(cell)
            # if cell has more than one possible tile
            if possible_tiles > 1 and is_collapsed:
                is_collapsed = False
            # if there is only one possible tile for the cell
            elif possible_tiles == 1:
                currently_done += 1
            # if there are no possible tiles for the cell
            elif not possible_tiles and is_valid:
                is_valid = False
        return is_collapsed, is_valid, currently_done

    def run(self) -> bool:
        """
        Runs WFC algorithm.

        Returns True if algorithm finishes successfully or False otherwise
        """
        # clear previously generated frames
        if self.animate:
            self.clear_frames()
        # main algorithm loop
        while True:
            is_collapsed, is_valid, currently_done = self.is_collapsed()

            if is_valid:
                if self.animate:
                    self.progress_bar("Generating frames:", currently_done)
                    # save animation frame
                    self.graphics(final=False)
                else:
                    self.progress_bar("Generating image:", currently_done)

            if not is_valid:
                return False
            if is_collapsed:
                break
            # make one algorithm iteration
            self.iterate()
        # save generated image
        self.graphics()
        if self.animate:
            # generate animation
            self.generate_animation()
        return True

    def calculate_avg_color(self, cell: list[int]) -> tuple[np.array]:
        """
        Calculates average color among all tiles in a given cell
        """
        avg_colors = []
        for tile_pos in cell:
            # calculate average color for every tile in the cell
            avg_colors.append(self.tile_set.tiles[tile_pos].avg_color)
        # calculate average color among all average colors of tiles converting it to int
        return tuple(np.array(avg_colors).mean(0).astype(int))

    def graphics(self, final=True) -> None:
        """
        Generates frame or final image for a WFC algorithm
        """
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
            # calculate max index in images names in folder
            max_index = self.chose_max_index(GENERATED_IMAGES, "result_")
            # save generated image with new max integer in name
            filename = f"{GENERATED_IMAGES}/result_{max_index + 1}.png"
            Image.fromarray(image).save(filename)
            print(f"Image is saved as {filename}")


def integer_input(title: str, min_: int, max_: int) -> int:
    """
    Prompts user to enter an integer in a given boundaries.

    If input is invalid, user will see corresponding Error and will be prompted to enter an integer again.

    If integer is valid, it returns from function.
    """
    while True:
        try:
            integer = int(input(title))
            if not min_ <= integer <= max_:
                print(f"Number should be between {min_} and {max_}")
                continue
            break
        except ValueError:
            print("Invalid input")
    return integer


def boolean_input(title: str) -> bool:
    """
    Prompts user to enter "y" or "n" which corresponds to True and False.

    If input is invalid, user will see corresponding Error and will be prompted to enter again.

    If input is valid, corresponds boolean returns from function.
    """
    while True:
        boolean = input(title).lower()
        if boolean == "n":
            return False
        elif boolean == "y":
            return True
        else:
            print("Invalid input")


def main():
    # choose tile set
    tile_set = chose_tile_set()

    while True:
        # input tile size
        tile_size = integer_input(
            f"Input tile size [min = {MIN_TILE_SIZE}, max = {MAX_TILE_SIZE}]: ",
            MIN_TILE_SIZE,
            MAX_TILE_SIZE
        )        # pars tiles with selected size
        try:
            tiles = pars_tiles(tile_set, tile_size)
            break
        except ValueError as e:
            print(str(e))

    # input if algorithm should use rotated versions of tiles
    rotate = boolean_input("Use rotated versions of tiles [y, n]: ")

    # input if algorithm should use mirrored versions of tiles
    mirror = boolean_input("Use mirrored versions of tiles [y, n]: ")

    # generate all possible tiles from parsed with rotations and mirrors
    tile_set = all_possible_tiles(tiles, rotate, mirror)

    # input if algorithm should use weights
    use_weights = boolean_input("Use weights when choosing tiles (recommended) [y, n]: ")

    # input output image width
    width = integer_input(
        f"Input width of the output image in tiles [min = {MIN_FIELD_WIDTH}, max = {MAX_FIELD_WIDTH}]: ",
        MIN_FIELD_WIDTH,
        MAX_FIELD_WIDTH
    )

    # input output image height
    height = integer_input(
        f"Input height of the output image in tiles [min = {MIN_FIELD_HEIGHT}, max = {MAX_FIELD_HEIGHT}]: ",
        MIN_FIELD_HEIGHT,
        MAX_FIELD_HEIGHT
    )

    # input if algorithm should generate animation
    animate = boolean_input("Generate animation (highly decreases performance) [y, n]: ")

    # input fps for animation
    fps = 0
    if animate:
        fps = integer_input(
            f"Input FPS for animation (~75 recommended) [min = {MIN_FPS}, max = {MAX_FPS}]: ",
            MIN_FPS,
            MAX_FPS
        )

    # input if algorithm should retry generating output if it wasn't able to
    tries = integer_input(
            f"Input number of tries for the algorithm to generate output [min = {MIN_RETRIES}, max = {MAX_RETRIES}]: ",
            MIN_RETRIES,
            MAX_RETRIES
        )

    counter = 1
    while True:
        # initialize WFC
        wfc = WFC(tile_set, use_weights, tile_size, width, height, animate, fps)
        start = time.perf_counter()
        # run WFC algorithm
        if wfc.run():
            print(f"WFC finished successfully in {str(datetime.timedelta(seconds=int(time.perf_counter() - start)))}")
            break
        else:
            print("\nCouldn't generate an image from a given tile set")
        if counter == tries:
            break
        else:
            counter += 1
            print(f"Attempt {counter}")


if __name__ == "__main__":
    main()
