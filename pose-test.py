# required libraries
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt



print(mp.__path__)
# Loading the image using OpenCV.
img = cv2.imread(r"D:\NIRAJ\CV_Pose\Niraj_Good.JPG")

# Getting the image's width and height.
img_width = img.shape[1]
img_height = img.shape[0]

# Creating a figure and a set of axes.
fig, ax = plt.subplots(figsize=(10, 10))
ax.axis('off')
ax.imshow(img[...,::-1])
plt.show()

# Initializing the Pose and Drawing modules of MediaPipe.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

with mp_pose.Pose(static_image_mode=True) as pose:
    """
    This function utilizes the MediaPipe library to detect and draw 'landmarks'
    (reference points) on an image. 'Landmarks' are points of interest
    that represent various body parts detected in the image.
    Args:
        static_image_mode: a boolean to inform if the image is static (True) or sequential (False).
    """
    # Make a copy of the original image.
    annotated_img = img.copy()
    # Processes the image.
    results = pose.process(img)
    # Set the circle radius for drawing the 'landmarks'.
    # The radius is scaled as a percentage of the image's height.
    circle_radius = int(.007 * img_height)
    # Specifies the drawing style for the 'landmarks'.
    point_spec = mp_drawing.DrawingSpec(color=(220, 100, 0), thickness=-1, circle_radius=circle_radius)
    # Draws the 'landmarks' on the image.
    mp_drawing.draw_landmarks(annotated_img,
                              landmark_list=results.pose_landmarks,
                              landmark_drawing_spec=point_spec)

# Make a copy of the original image.
annotated_img = img.copy()

# Specifies the drawing style for landmark connections.
line_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

# Draws both the landmark points and connections.
mp_drawing.draw_landmarks(
annotated_img,
landmark_list=results.pose_landmarks,
connections=mp_pose.POSE_CONNECTIONS,
landmark_drawing_spec=point_spec,
connection_drawing_spec=line_spec
)

# Select the coordinates of the points of interest.
l_elbow_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * img_width)
l_elbow_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * img_height)
l_elbow_z = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].z * img_width)

l_wrist_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * img_width)
l_wrist_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * img_height)
l_wrist_z = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].z * img_height)

l_shoulder_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * img_width)
l_shoulder_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * img_height)
l_shoulder_z = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z * img_height)

r_shoulder_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * img_width)
r_shoulder_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * img_height)
r_shoulder_z = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z * img_height)

l_hip_index_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * img_width)
l_hip_index_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * img_height)
l_hip_index_z = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z * img_height)

r_hip_index_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * img_width)
r_hip_index_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * img_height)
r_hip_index_z = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z * img_height)
# Print the coordinates on the screen.
print('Left elbow coordinates: (', l_elbow_x,',',l_elbow_y,'',l_elbow_z,')' )
print('Left wrist coordinates: (', l_wrist_x,',',l_wrist_y,',',l_wrist_z,')' )
print('Left shoulder coordinates: (', l_shoulder_x,',',l_shoulder_y,'',l_shoulder_z,')' )
print('Right shoulder coordinates: (', r_shoulder_x,',',r_shoulder_y,'',r_shoulder_z,')' )
print('Left hip index coordinates: (', l_hip_index_x,',',l_hip_index_y,',',l_hip_index_z,')' )
print('Right hip index coordinates: (', r_hip_index_x,',',r_hip_index_y,',',r_hip_index_z,')' )
# Displaying a graph with the selected points.
fig, ax = plt.subplots()
#ax.imshow(img[:, :, ::-1])
ax = plt.axes(projection='3d')
ax.plot([l_elbow_x, l_wrist_x, l_shoulder_x, l_hip_index_x, r_shoulder_x,r_hip_index_x], [l_elbow_y, l_wrist_y, l_shoulder_y, l_hip_index_y, r_shoulder_y,r_hip_index_y], [l_elbow_z,l_wrist_z,l_shoulder_z, l_hip_index_z, r_shoulder_z,r_hip_index_z], 'o')
ax.text(l_wrist_x, l_wrist_y, l_wrist_z, "LWrist", color='red')
ax.text(l_elbow_x, l_elbow_y, l_elbow_z, "Elbow", color='red')
ax.text(l_shoulder_x , l_shoulder_y , l_shoulder_z , "LSh", color='red')
ax.text(r_shoulder_x , r_shoulder_y , r_shoulder_z , "RSh", color='red')
ax.text(l_hip_index_x, l_hip_index_y, l_hip_index_z, "LHip", color='red')
ax.text(r_hip_index_x, r_hip_index_y, r_hip_index_z, "RHip", color='red')

ax.plot3D([l_shoulder_x,l_hip_index_x], [l_shoulder_y,l_hip_index_y], [l_shoulder_z,l_hip_index_z], 'r--')
ax.plot3D([l_shoulder_x,r_shoulder_x], [l_shoulder_y,r_shoulder_y], [l_shoulder_z,r_shoulder_z], 'r--')
ax.plot3D([r_hip_index_x,r_shoulder_x], [r_hip_index_y,r_shoulder_y], [r_hip_index_z,r_shoulder_z], 'r--')
ax.plot3D([r_hip_index_x,l_hip_index_x], [r_hip_index_y,l_hip_index_y], [r_hip_index_z,l_hip_index_z], 'r--')
ax.plot3D([l_shoulder_x,l_elbow_x], [l_shoulder_y,l_elbow_y], [l_shoulder_z,l_elbow_z], 'r--')
ax.plot3D([l_wrist_x,l_elbow_x], [l_wrist_y,l_elbow_y], [l_wrist_z,l_elbow_z], 'r--')

plt.show()