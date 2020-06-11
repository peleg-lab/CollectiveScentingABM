import os
import cv2
import sys
import glob2
import h5py
import shutil
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def read_config(base_exp_dir):
    cfg_path = glob2.glob(f"{base_exp_dir}/*.cfg")[0]

    with open(cfg_path, "r") as infile:
        lines = [line.split() for line in infile]
        cfg_opts = {}
        for key, val in lines:
            key = key.replace('--', '')

            try:
                val = float(val)
            except:
                try:
                    val = int(val)
                except:
                    if val.startswith("T"):
                        val = True
                    elif val.startswith("F"):
                        val = False
                    pass
            cfg_opts[key] = val
    return cfg_opts

def imgs2vid(imgs, outpath, fps=15):
    height, width, layers = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(outpath, fourcc, fps, (width, height), True)

    for img_i, img in enumerate(imgs):
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

def process_data(env_path, bee_path):
    # Get concentration maps
    with h5py.File(env_path, 'r') as infile:
        cmaps = np.array(infile['concentration'])

    # Min and max concentrations for heatmap
    min_c = np.min(cmaps)
    max_c = np.max(cmaps) * 0.85

    # Get bee measurements
    bee_data = {}
    with h5py.File(bee_path, 'r') as infile:
        for key, val in infile.items():
            bee_data[key] = np.array(val)
    bee_nums = np.unique(bee_data['bee_i'])
    bees = {}
    for bee_num in bee_nums:
        idxs = np.where(bee_data['bee_i']==bee_num)
        bee_x = bee_data['x'][idxs]
        bee_y = bee_data['y'][idxs]
        bee_state = bee_data['state'][idxs]
        distance = bee_data['distance_from_queen'][idxs]
        bias = bee_data['wx'][idxs], bee_data['wy'][idxs]
        bees[bee_num] = {"x" : bee_x, "y" : bee_y, "state": bee_state,
                        "distance": distance}

    return cmaps, min_c, max_c, bees

def plot_frame(cmaps, frame_i, min_c, max_c, bees, color_decoder, legend_colors, texts, script_config, convert_xy_to_index):
    # Process concentration map
    cmap = cmaps[frame_i]
    plt.imshow(cmap, cmap='Greens', vmin=min_c, vmax=max_c)
    clb = plt.colorbar(shrink=0.8, format='%.2f')
    clb.ax.set_title('C')

    # Process queen data
    queen = convert_xy_to_index(0)
    plt.scatter(queen, queen, c="red", s=100, edgecolors='black', marker='o')

    # Process worker data
    for bee_key, bee_vals in bees.items():
        x = bee_vals['x'][frame_i]
        y = bee_vals['y'][frame_i]
        state = bee_vals['state'][frame_i] + 1
        color = color_decoder[state]
        plt.scatter(convert_xy_to_index(x), convert_xy_to_index(y),
                    color=color, s=40, edgecolors='black')

    # Plot formatting
    patches = [ plt.plot([],[], marker="o", ms=10 if color_i==0 else 6, ls="", mec=None, color=legend_colors[color_i],
                markeredgecolor="black", label="{:s}".format(texts[color_i]) )[0]  for color_i in range(len(texts)) ]
    plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.15),
               loc='center', ncol=4, numpoints=1, labelspacing=0.3,
               fontsize='small', fancybox="True",
               handletextpad=0, columnspacing=0)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0, 300)
    plt.ylim(300, 0)

    # Title
    N = script_config['num_workers']
    T = script_config['worker_threshold']
    wb = int(script_config['worker_bias_scalar'])
    seed = int(script_config['random_seed'])
    title = f"Number of workers: {int(N)} -- Threshold: {T} \n$w_b$: {wb} -- Random seed: {seed}"
    plt.title(f"{title} \n t: {frame_i}/{time}")

    # Save frames
    file_path = f't{frame_i:05d}.png'
    filename = f'{MOVIE_FRAME_PATH}/{file_path}'
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()

def setup_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='N50_T0.01000_wb30.0_seed42', help='Path to experiment folder')
    parser.add_argument('-r', '--fps', type=int, default=5, help='FPS for output movie')
    parser.add_argument('-s', '--stepsize', type=int, default=1, help='Step size for plotting data')
    return parser.parse_args()

def main(trial_path, fps, stepsize):
    # Obtain parameters from config
    script_config = read_config(BASE_EXPERIMENT_DIR)
    X_MIN = script_config['x_min']
    X_MAX = script_config['x_max']
    DX = script_config['dx']
    GRID_SIZE = np.arange(X_MIN, X_MAX+DX, DX).shape[0]
    convert_xy_to_index = lambda xy: ((xy - X_MIN) / (X_MAX - X_MIN)) * GRID_SIZE

    # Get data paths
    env_path = os.path.join(BASE_EXPERIMENT_DIR, "envir_hist.h5")
    bee_path = os.path.join(BASE_EXPERIMENT_DIR, "bee_hist.h5")

    # Obtain & process data
    cmaps, min_c, max_c, bees = process_data(env_path, bee_path)

    # Make figure frames
    global time
    time = cmaps.shape[0]
    colors = ["red", "gray", "#479030", "orange", "gray", "red"]
    color_decoder = { i : color for i, color in enumerate(colors)}
    texts = ["Queen", "Random walk", "Scenting", "Directed walk"]
    legend_colors = ["red", "gray", "#479030", "orange"]

    for frame_i in range(0, time, stepsize):
        sys.stdout.write(f"\rMaking frame {frame_i}/{time}")
        sys.stdout.flush()

        plot_frame(cmaps, frame_i, min_c, max_c, bees, color_decoder, legend_colors,
                   texts, script_config, convert_xy_to_index)

    # Stitching frames together to create video
    all_img_paths = np.sort(glob2.glob(f"{MOVIE_FRAME_PATH}/*.png"))
    all_imgs = np.array([cv2.imread(img) for img in all_img_paths])
    savepath = f"{BASE_EXPERIMENT_DIR}/{trial_path}.mp4"
    imgs2vid(all_imgs, savepath, fps)

if __name__ == '__main__':
    opts = setup_opts()

    TRIAL_PATH = opts.path
    BASE_EXPERIMENT_DIR = f"experiments/{TRIAL_PATH}"
    MOVIE_FRAME_PATH = f"{BASE_EXPERIMENT_DIR}/movie_frames"
    if os.path.exists(MOVIE_FRAME_PATH):
        shutil.rmtree(MOVIE_FRAME_PATH)
    os.makedirs(MOVIE_FRAME_PATH, exist_ok=True)

    FPS = opts.fps

    INTERVAL = opts.stepsize

    print("\n---------- Visualizing bee model data ----------")
    main(TRIAL_PATH, FPS, INTERVAL)
    print("\nFin.\n")
