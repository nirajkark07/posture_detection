import mediapipe as mp
import cv2
from utils.plot_pose_live import *
import matplotlib.pyplot as plt
import math
import csv
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec

# Setup plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Setup mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
custom_style = mp_drawing_styles.get_default_pose_landmarks_style()
custom_connections = list(mp_pose.POSE_CONNECTIONS)

        # list of landmarks to exclude from the drawing
excluded_landmarks = [
    PoseLandmark.LEFT_EYE, 
    PoseLandmark.RIGHT_EYE, 
    PoseLandmark.LEFT_EYE_INNER, 
    PoseLandmark.RIGHT_EYE_INNER, 
    PoseLandmark.LEFT_EAR,
    PoseLandmark.RIGHT_EAR,
    PoseLandmark.LEFT_EYE_OUTER,
    PoseLandmark.RIGHT_EYE_OUTER,
    PoseLandmark.NOSE,
    PoseLandmark.MOUTH_LEFT,
    PoseLandmark.MOUTH_RIGHT]

for landmark in excluded_landmarks:
    # we change the way the excluded landmarks are drawn
    custom_style[landmark] = DrawingSpec(color=(0,0,0), thickness=None, circle_radius=0) 
    # we remove all connections which contain these landmarks
    custom_connections = [connection_tuple for connection_tuple in custom_connections 
                            if landmark.value not in connection_tuple]

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
            mp_drawing.draw_landmarks(image, results.pose_landmarks, connections = custom_connections, landmark_drawing_spec=custom_style)

        # draw 3D pose landmarks live
        plot_world_landmarks(ax, results.pose_world_landmarks)
        
        # draw image
        cv2.imshow("MediaPipePose", cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
