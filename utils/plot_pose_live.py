import matplotlib.pyplot as plt
from .math_utils import get_coordinates, get_data
import math
import csv

# connections for the MediaPipe topology
# LANDMARK_GROUPS = [
#     [11, 13, 15],  # left arm
#     [11, 23, 25, 27],  # left body side
#     [12, 14, 16],  # right arm
#     [12, 24, 26, 28],  # right body side
#     [11, 12],                      # shoulder
#     [23, 24],                      # waist
# ]

LANDMARK_GROUPS = [
    [11, 13, 15],  # left arm
    [12, 14, 16],  # right arm
    [13, 11, 23],  # left elbow, left shoulder, left knee
    [14, 12, 24],  # right elbow, right shoulder, right torso
    [11, 23, 25],  # left shoulder, left hip, left knee
    [12, 24, 26],  # right shoulder, right hip, right knee
    [23, 25, 27],  # left hip, left knee, left ankle
    [24, 26, 28],  # right hip, right knee, right ankle
    [24, 23], # left hip, right hip
    [12, 11] # left shoulder, right shoulder
]

csv_file = r'C:\RU\MASc\GIT\Pose_Model\angle_ranges.csv'

def plot_world_landmarks(ax, landmarks, landmark_groups=LANDMARK_GROUPS):
    """_summary_
    Args:
        ax: plot axes
        landmarks  mediapipe
    """

    # skip when no landmarks are detected
    if landmarks is None:
        return

    ax.cla()

    # had to flip the z axis
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(1, -1)

    # get coordinates for each group and plot
    for idx, group in enumerate(landmark_groups):
        plotX, plotY, plotZ, _ = get_coordinates(landmarks, group)
        ax.plot(plotX, plotZ, plotY, 'g')
        ax.scatter(plotX, plotZ, plotY, c='r', marker='o', s=10)

    plt.pause(.001)

def posture_color_data(landmarks, landmark_groups=LANDMARK_GROUPS):
    for idx, group in enumerate(landmark_groups):
        _, _, _, angD = get_coordinates(landmarks, group)
        idcx = get_data(csv_file, idx, angD)
        
        return idx, idcx

