import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from noise import snoise2
from scipy.ndimage import convolve
from skimage.morphology import disk

forest_size = 100
seed = random.randint(1, 1000)

tree_life = 50
fire_life = 10

custom_spread = False

if custom_spread:
    with open("spread.txt", "r") as f:
        spread_pattern = [[int(c) for c in l.strip()] for l in f.readlines()]
else:
    radius = 3

scale = 40

display_values = False

forest = np.zeros((forest_size, forest_size))

for i in range(forest_size):
    for j in range(forest_size):
        forest[i][j] = snoise2((i-forest_size/2) / scale, (j-forest_size/2) / scale, octaves=3, persistence=500, lacunarity=0.7, base=seed)
        

forest = np.round(1 + (tree_life - 1) * (forest - np.min(forest)) / (np.max(forest) - np.min(forest)))


def floor(forest: np.array):
    burning_cells = forest > -fire_life
    forest[burning_cells] -= 1
    return forest

def load(frame, forest: np.array, display_values: bool = False):
    colors = [(1, 0, 0), (1, 1, 0)]
    cmap_red_to_yellow = plt.cm.colors.LinearSegmentedColormap.from_list('RedToYellow', colors, N=256)
    colors = [(0, 1, 0), (0, 0, 1)]
    cmap_green_to_blue = plt.cm.colors.LinearSegmentedColormap.from_list('GreenToBlue', colors, N=256)

    custom_colors = ["dimgray" if display_values else "black"] + [cmap_red_to_yellow(int(50 + i * 200 / fire_life)) for i in range(fire_life)] + \
                    [cmap_green_to_blue(int(50 + i * 200 / tree_life)) for i in range(tree_life)]

    custom_cmap = plt.cm.colors.ListedColormap(custom_colors)

    plt.clf()
    plt.imshow(forest, cmap=custom_cmap, interpolation="nearest", vmin=-fire_life, vmax=tree_life)
    plt.title(f'Forest Fire Simulation - Frame {frame}')
    plt.axis('off')

    if display_values:
        for i in range(forest_size):
            for j in range(forest_size):
                plt.text(j, i, int(forest[i, j]), ha='center', va='center', color='white' if forest[i, j] > 0 else 'black')

    plt.pause(0.001)

def update(frame):
    global forest

    new_forest = forest.copy()

    ## for i in range(1, forest_size - 1):
    ##    for j in range(1, forest_size - 1):
    ##        if -fire_life < forest[i, j] <= 0:
    ##            new_forest[i - 1:i + 2, j - 1:j + 2] = floor(new_forest[i - 1:i + 2, j - 1:j + 2])

    ## [floor(new_forest[i - 1:i + 2, j - 1:j + 2]) for i in range(1, forest_size - 1) for j in range(1, forest_size - 1)
    ## if -fire_life < forest[i, j] <= 0]
    if custom_spread:
        kernel = np.array(spread_pattern)
    else:
        kernel = disk(radius)

    fire_cells = (-fire_life < forest) & (forest <= 0)

    try:
        neighborhood = convolve(fire_cells.astype(int), kernel, mode='constant', cval=0)
        new_forest[neighborhood > 0] = floor(new_forest[neighborhood > 0])
    except Exception as e:
        print(e)

    forest = new_forest

    load(frame, forest, display_values)

fig = plt.figure()

def initialize_fire(event):
    x, y = int(event.xdata), int(event.ydata)
    forest[y, x] = 0

fig.canvas.mpl_connect('button_press_event', initialize_fire)

ani = animation.FuncAnimation(fig, update, frames=2000, interval=10, repeat=False, cache_frame_data=True)

plt.show()