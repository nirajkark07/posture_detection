import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.plot_pose_live import plot_world_landmarks
from utils.math_utils import get_coordinates, get_data
from mediapipe import solutions as mp
from datetime import datetime, timedelta
from os.path import exists
import mediapipe as mp
import webbrowser
import cv2
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec

LANDMARK_GROUPS = [
    [11, 13, 15],  # left arm
    [12, 14, 16],  # right arm
    [13, 11, 23],  # left elbow, left shoulder, left hip
    [14, 12, 24],  # right elbow, right shoulder, right torso
    [11, 23, 25],  # left shoulder, left hip, left knee
    [12, 24, 26],  # right shoulder, right hip, right knee
    [23, 25, 27],  # left hip, left knee, left ankle
    [24, 26, 28],  # right hip, right knee, right ankle
    [24, 23], # left hip, right hip
    [12, 11], # left shoulder, right shoulder
    [16, 15], # left wrist, right wrist
    [28, 27] # left ankle, right ankle
]

class PoseVisualizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pose Visualizer")

        # Layout configuration to make video and plot side by side
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0,weight=1)

        # Create 3D plot on the left side
        self.create_3d_plot()

        #Initialize the live feed on the right side
        self.initialize_live_feed()

        # Posture slider and lights initialization
        self.initialize_posture_indicator()

        # Create a label for displaying the video feed
        self.video_label = tk.Label(self.root)
        self.video_label.grid(column=1, row=0, sticky="nsew")

        # Initialize all layouts
        self.create_initial_layout()
        self.create_views_layout()

        # Generate a unique filename for each session using a timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"angles_{timestamp}.csv"

        # Specify your directory path 
        directory_path = r"D:\NIRAJ\Pose_Model" 

        # If the directory doesn't exist, create it
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Create CSV file path within the specified directory
        self.csv_file_path = os.path.join(directory_path, filename)

        # Set last write time to None for timing checks
        self.last_write_time = None

        # Define the joint names
        self.joint_names = ['L_Elbow', 'L_Shoulder', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Wrist', 'R_Elbow', 'R_Shoulder', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Wrist']

        # Print the path where the CSV file will be saved
        print(f"The CSV file will be saved to: {self.csv_file_path}")


        # Initialize drawing ultility for MediaPipe pose landmarks
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.custom_style = self.mp_drawing_styles.get_default_pose_landmarks_style()
        self.custom_connections = list(self.mp_pose.POSE_CONNECTIONS)

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
            self.custom_style[landmark] = DrawingSpec(color=(0,0,0), thickness=None, circle_radius=0) 
            # we remove all connections which contain these landmarks
            self.custom_connections = [connection_tuple for connection_tuple in self.custom_connections 
                                    if landmark.value not in connection_tuple]

        self.pose = self.mp_pose.Pose(
            min_tracking_confidence=0.5,
            min_detection_confidence=0.5,
            model_complexity=1,
            smooth_landmarks=True,
        )

        # Write coords to CSV file every 5 seconds
        self.write_time = 5

        # CONTROL VARIABLES
        self.plotting_active = True # Initially True
        self.multiple_views_active = False # Initially False
        self.posture_active = False # Initially False

        # Create a start/stop button
        self.toggle_button = tk.Button(self.root, text="Stop", command=self.toggle_plot)
        self.toggle_button.grid(column=0, row=1, sticky="sw")  # Position the button at the bottom left

        # Create a View button
        self.views_button = tk.Button(self.root, text="Views", command=self.show_views)
        self.views_button.grid(column=1, row=1, sticky="sw")
        self.views_window = None

        # Create posture button
        self.posture_button = tk.Button(self.root, text="Posture", command=self.toggle_posture)
        self.posture_button.grid(column=2, row=1, sticky="sw")

        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset_timer)
        self.reset_button.grid(column=0, row=2, sticky="sw")  # Adjust the row and column as needed

        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_columnconfigure(2, weight=0)

        # Timer Variables
        self.start_time = None  # Keeps track of when the timer was started
        self.elapsed_time = timedelta(0)  # The amount of time passed since the start
        self.timer_running = False  # Indicates whether the timer is running

        # Create a timer label
        self.timer_label = tk.Label(self.root, text="00:00", font=("Arial", 16))
        self.timer_label.grid(column=2, row=0, sticky="ne", padx=5, pady=5)

        # Make sure the 3D plot does not start automatically
        self.plotting_active = False  # Initially False to keep the 3D graph stopped

        # Update the start/stop button text accordingly
        self.toggle_button.config(text="Start")

        # Start with the initial layout
        self.show_initial_layout()

        # Start main loop
        self.root.after(0, self.update)
        self.root.mainloop()

    def initialize_posture_indicator(self):
        self.posture_frame = tk.Frame(self.root)
        self.posture_frame.grid(column=0, row=2, columnspan=3, rowspan=6, sticky="nsew", padx=5, pady=5)
        self.posture_frame.grid_remove()  # Start with this frame hidden 

        # Helper function to create a label and lights for a specific joint
        def create_light_indicator(joint, row):
            label = tk.Label(self.posture_frame, text=joint)
            label.grid(row=row, column=0, sticky="e", padx=(10,2))
            lights = {}

            # Define the colors for the lights
            light_colors = ['red', 'yellow', 'green']
            light_bg_colors = ['light coral', 'light yellow', 'light green']

            for idx, color in enumerate(light_colors):
                canvas = tk.Canvas(self.posture_frame, width=20, height=20, bg=light_bg_colors[idx], highlightbackground='black', highlightthickness=1)
                # Use padx and pady to add some space between the squares
                canvas.grid(row=row, column=idx + 1, padx=2)
                lights[color] = canvas
            return lights

        # Adjust the column configuration for equal spacing
        for i in range(12):
            self.posture_frame.grid_rowconfigure(i, weight=1)
        self.posture_frame.grid_columnconfigure(list(range(4)), weight=1, uniform="group1")

        # Create light indicators for all joints
        self.left_elbow_lights = create_light_indicator("L_Elbow", 1)
        self.right_elbow_lights = create_light_indicator("R_Elbow", 2)
        self.left_shoulder_lights = create_light_indicator("L_Shoulder", 3)
        self.right_shoulder_lights = create_light_indicator("R_Shoulder", 4)
        self.left_hip_lights = create_light_indicator("L_Hip", 5)
        self.right_hip_lights = create_light_indicator("R_Hip", 6)
        self.left_knee_lights = create_light_indicator("L_Knee", 7)
        self.right_knee_lights = create_light_indicator("R_Knee", 8)

        # Create angle indicators for all joints
        self.angle_labels = {}
        for idx, joint in enumerate(["L_Elbow", "L_Shoulder", "L_Hip", "L_Knee", "R_Elbow", "R_Shoulder", "R_Hip", "R_Knee"]):
            angle_label = tk.Label(self.posture_frame, text="0°", bg="white", width=5, height=2, relief="solid", borderwidth=1)
            angle_label.grid(row=idx+1, column=4, padx=2, pady=2)
            self.angle_labels[joint] = angle_label

    def change_light_color(self, idx, idcx):

        lights_dict = {
            0: self.left_elbow_lights,
            2: self.left_shoulder_lights,
            4: self.left_hip_lights,
            6: self.left_knee_lights,
            1: self.right_elbow_lights,
            3: self.right_shoulder_lights,
            5: self.right_hip_lights,
            7: self.right_knee_lights,
            }

        if idx in lights_dict:
            lights = lights_dict[idx]

        # Update the colors of the lights based on the new color
        if idcx == '0':
            lights['green'].config(bg='green')
            lights['yellow'].config(bg='light yellow')
            lights['red'].config(bg='light coral')
        elif idcx == '1':
            lights['green'].config(bg='light green')
            lights['yellow'].config(bg='yellow')
            lights['red'].config(bg='light coral')
        elif idcx == '2':
            lights['green'].config(bg='light green')
            lights['yellow'].config(bg='light yellow')
            lights['red'].config(bg='red')

    def toggle_posture(self):
        self.posture_active = not self.posture_active # Initially set False, set to True
        if self.posture_active:
            self.posture_frame.grid()
            self.posture_button.config(text="Hide Posture")
        else:
            self.posture_frame.grid_remove() # Unhide the posture
            self.posture_button.config(text="Posture")

    def create_initial_layout(self):
        # Layout configuration for the initial layout
        self.root.grid_columnconfigure(0, weight=1, minsize=400)  # Video feed column
        self.root.grid_columnconfigure(1, weight=1, minsize=400)  # 3D plot column
        self.root.grid_rowconfigure(0, weight=1)

    def create_3d_plot(self):
        # Create the 3D plot
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.draw()
    
        # Place the canvas widget for the 3D plot
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(column=1, row=0, sticky="nsew", padx=5, pady=5)

    def create_views_layout(self):
        # Ensure the views layout uses an additional frame for easier management
        self.views_frame = tk.Frame(self.root)
        self.views_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.views_frame.grid_remove()  # Initially hide this frame

        # Configure the views frame grid layout
        self.views_frame.grid_columnconfigure(0, weight=1)
        self.views_frame.grid_columnconfigure(1, weight=1)
        self.views_frame.grid_rowconfigure(0, weight=0)  # Label row, no need to expand
        self.views_frame.grid_rowconfigure(1, weight=1)  # Plot row, should expand

        # LABELS FOR EACH VIEW
        top_view_label = tk.Label(self.views_frame, text="Transverse Plane", font=("Arial", 12))
        top_view_label.grid(column=0, row=0, sticky="nsew")

        front_view_label = tk.Label(self.views_frame, text="Coronal Plane", font=("Arial", 12))
        front_view_label.grid(column=0, row=2, sticky="nsew")

        side_view_label = tk.Label(self.views_frame, text="Sagittal Plane", font=("Arial", 12))
        side_view_label.grid(column=1, row=2, sticky="nsew")  # Position adjusted for the layout

        plot_view_label = tk.Label(self.views_frame, text="3D Plot", font=("Arial", 12))
        plot_view_label.grid(column=1, row=0, sticky="nsew")  # Position adjusted for the layout

        # Creating and embedding matplotlib figures for top, front, side, and additional 3D plots
        self.top_view_fig, self.top_view_ax = self.create_plot_for_view()
        self.embed_plot_in_frame(self.top_view_fig, 0, 1)

        self.front_view_fig, self.front_view_ax = self.create_plot_for_view()
        self.embed_plot_in_frame(self.front_view_fig, 0, 3)

        self.side_view_fig, self.side_view_ax = self.create_plot_for_view()
        self.embed_plot_in_frame(self.side_view_fig, 1, 3)  # Adjust row for layout

        self.additional_3d_fig, self.additional_3d_ax = self.create_plot_for_view()
        self.embed_plot_in_frame(self.additional_3d_fig, 1, 1)  # Adjust row for layout

    def create_plot_for_view(self):
        # Create a matplotlib plot for a view
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        return fig, ax

    def embed_plot_in_frame(self, fig, col, row):
        # Embed a plot into the specified location in the views_frame
        canvas = FigureCanvasTkAgg(fig, master=self.views_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(column=col, row=row, sticky="nsew", padx=5, pady=5)

    def show_initial_layout(self):
        # Show the video label and the canvas for the initial layout
        self.video_label.grid(column=1, row=0, sticky="nsew")
        self.canvas.get_tk_widget().grid(column=0, row=0, sticky="nsew")

        # Hide the views layout frame
        self.views_frame.grid_remove()
        self.current_layout = "initial" # self.current_layout is what changes the views
        self.views_button.config(text="Views")

    def show_views_layout(self):
        # Hide the video label and the canvas for the 3D plot
        self.video_label.grid_remove()
        self.canvas.get_tk_widget().grid_remove()

        # Show the views layout frame
        self.views_frame.grid()
        self.current_layout = "views"
        self.views_button.config(text="Home")

    def toggle_views(self):
        # Toggle between the initial layout and the views layout
        if self.current_layout == "initial":
            self.show_views_layout()
            self.current_layout = "views"
        else:
            self.show_initial_layout()
            self.current_layout = "initial"

    def show_views(self):
        # Need to add control flag
        self.toggle_views()
        self.multiple_views_active = not self.multiple_views_active # Initially False set to True
        self.plotting_active = not self.plotting_active # Initially False set to True
        # Update the text of the views button to reflect the current mode
        self.views_button.config(text="Home" if self.current_layout == "views" else "Views")   
    
    def update_views(self):
        if self.plotting_active and self.latest_landmarks:
            self.update_single_view(self.top_view_ax, self.top_view_fig, 'top')
            self.update_single_view(self.front_view_ax, self.front_view_fig, 'front')
            self.update_single_view(self.side_view_ax, self.side_view_fig, 'side')
            self.update_single_view(self.additional_3d_ax, self.additional_3d_fig, '3d')
            self.update_live_feed()

    def update_single_view(self, ax, fig, view_type):
        ax.cla()  # Clear the current axes
        plot_world_landmarks(ax, self.latest_landmarks, LANDMARK_GROUPS)  # Use your own function here

        if view_type == "top":
            ax.view_init(elev=90, azim=-90)
        elif view_type == "front":
            ax.view_init(elev=180, azim=-90)
        elif view_type == "side":
            ax.view_init(elev=180, azim=0)
        elif view_type == "3d":
            ax.view_init(elev=180, azim=-45)
            
        # Set static limits for axes
        ax.set_xlim3d([-1, 1])
        ax.set_ylim3d([-1, 1])
        ax.set_zlim3d([-1, 1])

        fig.canvas.draw_idle()

    def prepare_live_feed_view(self):
        self.live_feed_label = tk.Label(self.views_window)
        self.live_feed_label.grid(column=1, row=1, sticky="nsew")
        self.update_live_feed()

    def update_live_feed(self):
        if self.views_window is None or not self.views_window.winfo_exists():
            return
        success, image = self.cap.read()
        if success:
            cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(cv_image)
            im_pil = im_pil.resize((200, 150), Image.Resampling.LANCZOS)
            image_tk = ImageTk.PhotoImage(image=im_pil)
            self.live_feed_label.configure(image=image_tk)
            self.live_feed_label.image = image_tk
        self.views_window.after(100, self.update_live_feed)        

    def initialize_live_feed(self):
        # Open webcam
        self.cap=cv2.VideoCapture(2)
        # Set the resolution of the webcam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Displaying video feed
        self.video_label = tk.Label(self.root)
        self.video_label.grid(column=1, row=0, sticky="nsew")

    def toggle_plot(self):
        self.plotting_active = not self.plotting_active
        self.toggle_button.config(text="Start" if not self.plotting_active else "Stop")

        if self.plotting_active:
            self.start_timer()
            self.save_coordinates_csv = True # Set true flag
            self.generate_blank_csv() # Generate blank csv.
        else:
            self.stop_timer()
            self.save_coordinates_csv = False

    def start_timer(self):
        if not self.timer_running:
            self.start_time = datetime.now() - self.elapsed_time
            self.timer_running = True
            self.update_timer()

    def stop_timer(self):
        if self.timer_running:
            self.timer_running = False
            self.elapsed_time = datetime.now() - self.start_time

    def update_timer(self):
        if self.timer_running:
            # Calculate elapsed time
            self.elapsed_time = datetime.now() - self.start_time
            # Update the timer label
            self.timer_label.config(text=self.format_timedelta(self.elapsed_time))
            # Schedule the next timer update
            self.root.after(1000, self.update_timer)

    def start_timer(self):
        if not self.timer_running:
            # Start from the previous elapsed time
            self.start_time = datetime.now() - self.elapsed_time
            self.timer_running = True
            self.update_timer()

    def stop_timer(self):
        if self.timer_running:
            # Stop the timer and capture the elapsed time
            self.timer_running = False
            self.elapsed_time = datetime.now() - self.start_time

    def reset_timer(self):
        # Stop the timer if it is running
        if self.timer_running:
            self.stop_timer()
        # Reset the elapsed time to zero
        self.elapsed_time = timedelta(0)
        # Update the timer label to reflect the reset
        self.timer_label.config(text="00:00")

    @staticmethod
    def format_timedelta(td):
        # Format the time delta to minutes and seconds
        minutes, seconds = divmod(td.seconds, 60)
        return "{:02}:{:02}".format(minutes, seconds)       

    def write_angles_to_csv(self, angles):
        try:
            file_exists = exists(self.csv_file_path)
            with open(self.csv_file_path, 'a', newline='') as csvfile:
                # fieldnames = ['Time (s)', 'L_Elbow', 'L_Shoulder', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Wrist', 'R_Elbow', 'R_Shoulder', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Wrist']
                fieldnames = ['Time (s)', 'L_Elbow_x', 'L_Elbow_y', 'L_Elbow_z', 'L_Shoulder', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Wrist', 'R_Elbow', 'R_Shoulder', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Wrist']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(angles)
                csvfile.flush()
                csvfile.close()
            print(f"Data written to {self.csv_file_path}")  # Log success
        except Exception as e:
            print(f"Failed to write data: {e}")  # Log any exceptions

    ## GENEREATE BLANK CSV FILE.
    def generate_blank_csv(self):
        now = datetime.now()
        self.filename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
        with open(self.filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            header = ["Time (s)",
                      "L_Elbow_x", "L_Elbow_y", "L_Elbow_z",
                      "L_Shoulder_x", "L_Shoulder_y", "L_Shoulder_z",
                      "L_Hip_x", "L_Hip_y", "L_Hip_z",
                      "L_Knee_x", "L_Knee_y", "L_Knee_z",
                      "R_Elbow_x", "R_Elbow_y", "R_Elbow_z",
                      "R_Shoulder_x", "R_Shoulder_y", "R_Shoulder_z",
                      "R_Hip_x", "R_Hip_y", "R_Hip_z",
                      "R_Knee_x", "R_Knee_y", "R_Knee_z",]
            csvwriter.writerow(header)

    def create_3d_plot(self):
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(column=0, row=0, sticky="nsew")

    def update(self):
        success, image = self.cap.read()  # Capture frame-by-frame

        if not success:
            self.root.after(10, self.update)
            return

        # Convert the BGR image to RGB.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Calculate pose using MediaPipe on the captured frame.
        results = self.pose.process(rgb_image)

        # Update the video label with the new image regardless of plotting state
        if results.pose_landmarks:
            #TODO: Clean up code and add better comments.
            results.pose_landmarks.landmark[24].y -= 0.025         
            results.pose_landmarks.landmark[23].y -= 0.025
            results.pose_landmarks.landmark[24].x -= 0.015
            results.pose_landmarks.landmark[23].x += 0.015
            # Draw landmarks on the RGB image if plotting is active
            if self.plotting_active:
                self.mp_drawing.draw_landmarks(
                    rgb_image, 
                    results.pose_landmarks, 
                    connections=self.custom_connections, 
                    landmark_drawing_spec=self.custom_style
                )
            
        # Convert the RGB image to a Tkinter PhotoImage and update the label
        im_pil = Image.fromarray(rgb_image)
        image_tk = ImageTk.PhotoImage(image=im_pil)
        self.video_label.configure(image=image_tk)
        self.video_label.image = image_tk

        # Update 3D plot with new pose landmarks if plotting is active
        if self.plotting_active and results.pose_world_landmarks:
            self.latest_landmarks = results.pose_world_landmarks

            if self.multiple_views_active:
                self.update_views()
            self.update_3d_plot(results.pose_world_landmarks)

            if self.posture_active:
                self.update_posture_indicators(results.pose_world_landmarks)
            # New function to save (x,y,z) location and posture indicator foe each.
            if self.save_coordinates_csv:
                print("hello")
                self.write_coords_to_csv(results.pose_world_landmarks)
        self.root.after(10, self.update)

    def update_posture_indicators(self, landmarks):
        current_time = datetime.now()
        angles_to_write = {}
        
        # Calculate angles for each joint
        for idx, group in enumerate(LANDMARK_GROUPS):
            csv_file = r'D:\NIRAJ\Pose_Model\angle_ranges.csv'
            _, _, _, angD = get_coordinates(landmarks, group)
            idcx = get_data(csv_file, idx, angD)
            if idcx is None:
                continue

            self.change_light_color(idx, idcx)
            # joint_name = ["L_Elbow", "L_Shoulder", "L_Hip", "L_Knee", "L_Ankle", "L_Wrist", "R_Elbow", "R_Shoulder", "R_Hip", "R_Knee", "R_Ankle", "R_Wrist"][idx]
            joint_name = ["L_Elbow", "L_Shoulder", "L_Hip", "L_Knee", "R_Elbow", "R_Shoulder", "R_Hip", "R_Knee"][idx]
            angles_to_write[joint_name] = angD  # Add angle to the dictionary for CSV writing

            angle_label = self.angle_labels.get(joint_name)
            if angle_label:
                angle_label.config(text=f"{180 - angD:.2f}°")
        
        # # Write angles to CSV every 5 seconds
        # if self.last_write_time is None or (current_time - self.last_write_time).total_seconds() >= 5:
        #     if self.plotting_active and angles_to_write:
        #         # Add the elapsed time to the dictionary
        #         angles_to_write['Time (s)'] = 5 * round(self.elapsed_time.total_seconds() / 5)
        #         self.write_angles_to_csv(angles_to_write)
        #         self.last_write_time = current_time
    
    def write_coords_to_csv(self, results):
        current_time = datetime.now()
        if self.last_write_time is None or (current_time - self.last_write_time).total_seconds() >= self.write_time: 
            time = 5 * round(self.elapsed_time.total_seconds() / 5)
            with open(self.filename, 'a') as outfile:
                writer = csv.writer(outfile)
                writer.writerow([time, results.landmark[13].x, results.landmark[13].y, results.landmark[13].z])
                self.last_write_time = current_time


    def update_main_live_feed(self, image, results):                                            # THIS IS WHERE WE ARE UPDATING THE 2D plot
        # Convert the BGR image to RGB, then to a Tkinter PhotoImage
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if results.pose_landmarks:
            # Optionally draw landmarks on the rgb_image before converting it to PhotoImage
            self.mp_drawing.draw_landmarks(
                rgb_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )
        im_pil = Image.fromarray(rgb_image)
        image_tk = ImageTk.PhotoImage(image=im_pil)
        self.video_label.configure(image=image_tk)
        self.video_label.image = image_tk

    def on_views_window_close(self):
                                                                        # remove self.views_window_close.destroy()
        self.close_views_window()

    def update_3d_plot(self, landmarks):
        self.ax.cla()

        # had to flip the z axis
        self.ax.set_xlim3d(-1, 1)
        self.ax.set_ylim3d(-1, 1)
        self.ax.set_zlim3d(-1, 1)

        plot_world_landmarks(self.ax, landmarks)

        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    gui = PoseVisualizerGUI(root)

    # spacing of the lights 
    # see a working GUI by next wednesday on home page add timer and tie it to the start stop button change the function to have the the plot at a stop
    # csv file, Time (every 5 seconds )

    # add wrist and ankles 