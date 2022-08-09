# How to set up and run project

On Windows:
1. Download `Python 3.10` or above
2. Execute `setup.bat` once to create virtual environment and download all required libraries into it
3. Execute `run.bat` to run project
   
On other OS:
1. Download `Python 3.10` or above
2. Create virtual environment and download all required libraries into it 
3. Run `main.py` using created virtual environment

# About this implementation

You can view original WaveFunctionCollapse repository [here](https://github.com/mxgmn/WaveFunctionCollapse). 

There are three main classes in this implementation.

1. Tile 
    - stores tile (an N x N image represented in an array of pixels)
    - stores 4 sides of a tile (top, right, bottom, left) hashed to an integer
    - stores average color of a tile for animation purposes
    - stores weight of a tile to make sure that tile distribution in an output image would be the same for every tile
2. TileSet 
   - stores all unique tiles
   - has method of adding tiles to itself and increasing tiles weight
3. WFC
   - stores TileSet object 
   - stores grid of positions of tiles in a tile set in a form of a list of lists of positions
   - and other necessary variables for the algorithm
   - has main logic of an algorithm

