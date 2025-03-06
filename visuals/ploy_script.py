import os.path as osp  # Path handling module
from Motion import BVH  # Custom BVH file processing module
from Motion.transforms import quat2repr6d  # Convert quaternion to 6D rotation representation
from glob import glob  # File path matching module
from Motion.Animation import positions_global as anim_pos  # Global position calculation module
from Motion.Animation import Animation  # Animation data structure
from Motion.Quaternions import Quaternions  # Quaternion processing class
from Motion.transforms import repr6d2quat  # Convert 6D rotation representation to quaternion
from Motion.AnimationStructure import get_kinematic_chain  # Get kinematic chain structure
from os.path import join as pjoin
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from bokeh.colors.named import orange
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from itertools import cycle
# import cv2
from textwrap import wrap


def list_cut_average_sinmdm(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[]):
    """
    A wrapper around explicit_plot_3d_motion that uses gt_frames to determine the colors for the frames.
    """
    matplotlib.use('Agg')
    data = joints.copy().reshape(len(joints), -1, 3)
    frames_number = data.shape[0]
    frame_colors = ['blue' if index in gt_frames else 'orange' for index in range(frames_number)]
    if dataset in ['mixmamo', 'bvh_general']:
        frame_colors = ['dragon'] * frames_number
    if vis_mode == 'unfold':  # FIXME: hard coded intervals
        frame_colors = ['purple'] * 40 + ['orange'] * 40
        frame_colors = ['orange'] * 80 + frame_colors * 1024
    explicit_plot_3d_motion(
        save_path, kinematic_tree, joints, title, dataset,
        figsize=figsize, fps=fps, radius=radius, vis_mode=vis_mode, frame_colors=frame_colors
    )


def explicit_plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3, vis_mode="default", frame_colors=[]):
    """
    Outputs the 3D motion to an MP4 file.
    """
    matplotlib.use("Agg")

    # Format title text for display
    if isinstance(title, str):
        title = ["\n".join(wrap(title, 20))]
    elif isinstance(title, list):
        title = ["\n".join(wrap(t, 20)) for t in title]

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3.0, radius * 2 / 3.0])
        fig.suptitle(title[0], fontsize=10)
        ax.grid(False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        # Plot a plane in the XZ direction
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    data = joints.copy().reshape(len(joints), -1, 3)
    # data[..., [1, 2]] = data[..., [2, 1]]  # Swap y and z axes

    # Data preparation for specific datasets
    if dataset == "humanml":
        data *= 1.3  # Scale for visualization
    elif dataset in ['humanact12', 'uestc', 'amass']:
        data *= -1.5  # Reverse axes and scale
    elif dataset == 'babel':
        data *= -1.3
    elif dataset in ['mixmamo', 'bvh_general']:
        data *= 20

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    plt.tight_layout()
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors_purple = ["#6B31DB", "#AD40A8", "#AF2B79", "#9B00FF", "#D836C1"]

    colors_dragon = ["#B1B1B1", "#B75A39", "#d59076", "#DD5A37", "#DD5A37", "#DD5A37",
                     "#D69E00", "#D69E00", "#D69E00", "#d59076", "#d59076", "#d59076",
                     "#d59076", "#d59076", "#FF6D00", "#FF6D00", "#FF6D00", "#FF6D00", "#FF6D00",
                     "#FF6D00", "#FF6D00", "#FF6D00", "#DDB50E", "#DDB50E", "#DDB50E", "#DDB50E", "#DDB50E",
                     "#DDB50E", "#DDB50E", "#DDB50E", "#cf8063", "#cf8063", "#cf8063",
                     "#b73838", "#b73838", "#b73838", "#b73838", "#b73838", "#9c4d30"]

    colors_upper_body = colors_blue[:2] + colors_orange[2:]
    colors_dict = {
        "blue": colors_blue,
        "orange": colors_orange,
        "purple": colors_purple,
        "upper_body": colors_upper_body,
        "dragon": colors_dragon
    }

    colors = colors_orange
    if vis_mode == 'upper_body':
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'lower_body':
        colors[2] = colors_blue[2]
        colors[3] = colors_blue[3]
        colors[4] = colors_blue[4]
    elif vis_mode == 'gt':
        colors = colors_blue
    if dataset in ['bvh_general']:
        colors = colors_dragon

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        ax.clear()  # Clear the axes for new plot
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        if len(title) > 1:
            fig.suptitle(title[index], fontsize=10)
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])
        used_colors = colors_dict[frame_colors[index]] if index < len(frame_colors) else colors_dict["blue"]
        for i, (chain, color) in enumerate(zip(kinematic_tree, cycle(used_colors))):
            linewidth = 2.0
            if dataset in ['mixmamo', 'bvh_general']:
                linewidth = 2.0
            else:
                linewidth = 4.0 if i < 5 else 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 // fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()


if __name__ == "__main__":
    in_dir = f'/root/autodl-tmp/PCMRL/output/epoch_50000/bvh/predict'
    bvh_files = glob(osp.join(in_dir, '*.bvh'))
    save_dir = "./output/epoch_50000/bvh_vis/"
    fps = 20
    dataset = "bvh_general"

    for idx, in_file in enumerate(bvh_files):
        anim, joint_names, frametime = BVH.load(in_file)
        kinematic_chain = get_kinematic_chain(anim.parents)
        print('Pose shape:', anim.positions.shape)
        joint = anim_pos(anim)
        save_path = pjoin(save_dir, '%02d.mp4' % (idx))
        plot_3d_motion(save_path, kinematic_chain, joints=joint, dataset=dataset, title='%02d' % (idx), fps=fps)
