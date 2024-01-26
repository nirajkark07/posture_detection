# required libraries
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Find the markers on the body
def find_markers(img):
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
        results = pose.process(annotated_img)
        # Set the circle radius for drawing the 'landmarks'.
    
    return results

def get_coords(results, landmark, img_width, img_height):
    if results.pose_landmarks is not None:
        x = int(results.pose_landmarks.landmark[landmark].x*img_width)
        y = int(results.pose_landmarks.landmark[landmark].y*img_height)
        z = int(results.pose_landmarks.landmark[landmark].z*img_width)
    
        return [x, y, z]
    else:
        return[0, 0, 0]

def draw_plot(pose_matrix, fig, sc):
    if pose_matrix is not None:
        sc._offsets3d = (pose_matrix[:, 0], pose_matrix[:, 1], pose_matrix[:, 2])

        l_shoulder = pose_matrix[2,:]
        l_hip = pose_matrix[4,:]
        line = ax.plot3D(l_shoulder, l_hip, 'r--') 
        l = line.pop
        
        #Left Shoulder -> Left Hip
        # ax.plot3D([l_shoulder_x,r_shoulder_x], [l_shoulder_y,r_shoulder_y], [l_shoulder_z,r_shoulder_z], 'r--') # Left Shoulder ->Right Shoulder
        # ax.plot3D([r_hip_index_x,r_shoulder_x], [r_hip_index_y,r_shoulder_y], [r_hip_index_z,r_shoulder_z], 'r--') #Right Shoulder -> R Hip
        # ax.plot3D([r_hip_index_x,l_hip_index_x], [r_hip_index_y,l_hip_index_y], [r_hip_index_z,l_hip_index_z], 'r--') # R Hip -> L Hip
        # ax.plot3D([l_shoulder_x,l_elbow_x], [l_shoulder_y,l_elbow_y], [l_shoulder_z,l_elbow_z], 'r--') #Left Shoulder -> Left Elbow
        # ax.plot3D([l_wrist_x,l_elbow_x], [l_wrist_y,l_elbow_y], [l_wrist_z,l_elbow_z], 'r--') #Left Wrist -> Left Elbow
        # ax.plot3D([r_shoulder_x,r_elbow_x], [r_shoulder_y,r_elbow_y], [r_shoulder_z,r_elbow_z], 'r--') #Right Shoulder -> Right Elbow
        # ax.plot3D([r_wrist_x,r_elbow_x], [r_wrist_y,r_elbow_y], [r_wrist_z,r_elbow_z], 'r--') #Right Wrist -> Right Elbow
        # ax.plot3D([r_hip_index_x,r_knee_x], [r_hip_index_y,r_knee_y], [r_hip_index_z,r_knee_z], 'r--') #Right Knee -> Right Hip
        # ax.plot3D([l_knee_x,l_hip_index_x], [l_knee_y,l_hip_index_y], [l_knee_z,l_hip_index_z], 'r--') #Left Knee -> Left Hip
        # ax.plot3D([r_knee_x,r_ankle_x], [r_knee_y,r_ankle_y], [r_knee_z,r_ankle_z], 'r--') #Right Knee -> Right Ankle
        # ax.plot3D([l_knee_x,l_ankle_x], [l_knee_y,l_ankle_y], [l_knee_z,l_ankle_z], 'r--') #Left Knee -> Left Ankle
        # ax.plot3D([nose_x], [nose_y], [nose_z], 'b') #Nose

        fig.canvas.draw()
        fig.canvas.flush_events()
        l.remove()
        del l
    

        return sc
    
def initialize_plot(pose_matrix, scatter, ax):
    if scatter == False and pose_matrix is not None:
        scatter = True
        sc = ax.scatter(pose_matrix[0:,0], pose_matrix[0:,1], pose_matrix[0:,2])

        return scatter, sc



# define a video capture object 
cap = cv2.VideoCapture(0) 
cap.set(3,800)
cap.set(4,800)

# Img data
img_width = 800
img_height = 800

# Initialize Graph
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = False

while(True): 
      
    ret, frame = cap.read() # Reads the frame

    results = find_markers(frame) # GET DETECTIONS

    # PARSE COORDINATES
    l_elbow = get_coords(results, mp_pose.PoseLandmark.LEFT_ELBOW, img_width, img_height)
    r_elbow= get_coords(results, mp_pose.PoseLandmark.RIGHT_ELBOW, img_width, img_height)
    l_shoulder = get_coords(results, mp_pose.PoseLandmark.LEFT_SHOULDER, img_width, img_height)
    r_shoulder = get_coords(results, mp_pose.PoseLandmark.RIGHT_SHOULDER, img_width, img_height)
    l_hip = get_coords(results, mp_pose.PoseLandmark.LEFT_HIP, img_width, img_height)
    r_hip = get_coords(results, mp_pose.PoseLandmark.RIGHT_HIP, img_width, img_height)
    
    l_wrist = get_coords(results, mp_pose.PoseLandmark.LEFT_WRIST, img_width, img_height)
    r_wrist = get_coords(results, mp_pose.PoseLandmark.RIGHT_WRIST, img_width, img_height)
    l_knee = get_coords(results, mp_pose.PoseLandmark.LEFT_KNEE, img_width, img_height)
    r_knee = get_coords(results, mp_pose.PoseLandmark.RIGHT_KNEE, img_width, img_height)
    l_ankle = get_coords(results, mp_pose.PoseLandmark.LEFT_ANKLE, img_width, img_height)
    r_ankle = get_coords(results, mp_pose.PoseLandmark.RIGHT_ANKLE, img_width, img_height)

    nose = get_coords(results, mp_pose.PoseLandmark.NOSE, img_width, img_height)
    #neck = centroid nose and shoulder
    pose_matrix = np.array([l_elbow, r_elbow, l_shoulder, r_shoulder, l_hip, r_hip, l_wrist, r_wrist, l_knee, r_knee, l_ankle, r_ankle, nose])
    
    # UPDATE PLOT
    if scatter == False:
        scatter, sc = initialize_plot(pose_matrix, scatter, ax)
    else:
        sc = draw_plot(pose_matrix, fig, sc)
  
    #Display live video feed
    cv2.imshow('frame', frame)
    
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release theq cap object 
cap.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 

#####Need to apply MP to live video, need to corelate xyz points for live plot



              