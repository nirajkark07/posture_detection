import mediapipe as mp
import cv2
from utils.plot_pose_live import *
import matplotlib.pyplot as plt
import math
import csv

# Setup plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Setup mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Open webcam
cap = cv2.VideoCapture(0) # change index

with mp_pose.Pose(
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5,
    model_complexity=1,
    smooth_landmarks=True,
) as pose:
    while cap.isOpened():
        # read webcam image
        success, image = cap.read()

        # skip empty frames
        if not success:
            continue

        # calculate pose
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # draw 3D pose landmarks live
        plot_world_landmarks(ax, results.pose_world_landmarks)

        # draw 2D graph of the arm joint
        # plot_2d_arm_joint(ax2, results.pose_world_landmarks, landmark_groups=LANDMARK_GROUPS)
        
        # draw image
        cv2.imshow("MediaPipePose", cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
