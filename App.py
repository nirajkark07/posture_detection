import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.plot_pose_live import plot_world_landmarks
from utils.math_utils import get_coordinates, get_data
from mediapipe import solutions as mp
import mediapipe as mp
import cv2
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec

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

        # CONTROL VARIABLES
        self.plotting_active = True # Initially True
        self.multiple_views_active = False # Initially False
        self.posture_active = False # Initially False

        # Create a start/stop button
        self.toggle_button = tk.Button(self.root, text="Stop", command=self.toggle_plot)
        self.toggle_button.grid(column=0, row=2, sticky="sw")  # Position the button at the bottom left

        # Create a View button
        self.views_button = tk.Button(self.root, text="Views", command=self.show_views)
        self.views_button.grid(column=0, row=4, sticky="sw")
        self.views_window = None

        # Create posture button
        self.posture_button = tk.Button(self.root, text="Posture", command=self.toggle_posture)
        self.posture_button.grid(column=0, row=6, sticky="sw")

        # Start with the initial layout
        self.show_initial_layout()

        # Start main loop
        self.root.after(0, self.update)
        self.root.mainloop()

    def initialize_posture_indicator(self):
        self.posture_frame = tk.Frame(self.root)
        self.posture_frame.grid(column=2, row=0, rowspan=6, sticky="nsew", padx=5, pady=5)
        self.posture_frame.grid_remove()  # Start with this frame hidden 

        # Define the colors for the lights
        light_colors = ['green', 'yellow', 'red']
        light_bg_colors = ['light green', 'light yellow', 'light coral']

        # Helper function to create a label and lights for a specific joint
        def create_light_indicator(joint, column):
            label = tk.Label(self.posture_frame, text=joint)
            label.grid(column=column, row=0, sticky="s", padx=5)
            lights = {}
            for idx, color in enumerate(light_colors):
                canvas = tk.Canvas(self.posture_frame, width=20, height=20, bg=light_bg_colors[idx], highlightbackground='black', highlightthickness=1)
                # Use padx and pady to add some space between the squares
                canvas.grid(column=column, row=idx + 1, padx=5, pady=2)
                lights[color] = canvas
            return lights

        # Adjust the column configuration for equal spacing
        self.posture_frame.grid_columnconfigure(0, weight=1)
        for i in range(1, 5):
            self.posture_frame.grid_columnconfigure(i, weight=1)

        # Adjust the row configuration so all the lights have equal space
        for i in range(4):
            self.posture_frame.grid_rowconfigure(i, weight=1)

        # Create light indicators for all joints
        self.elbow_lights = create_light_indicator("Elbow", 1)
        self.shoulder_lights = create_light_indicator("Shoulder", 2)
        self.hip_lights = create_light_indicator("Hip", 3)
        self.knee_lights = create_light_indicator("Knee", 4)

    def change_light_color(self, idx, idcx):

        lights_dict = {
            0: self.elbow_lights,
            2: self.shoulder_lights,
            4: self.hip_lights,
            6: self.knee_lights
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
        if self.posture_frame.winfo_ismapped():
            self.posture_frame.grid_remove()
            self.posture_button.config(text="Posture")
        else:
            self.posture_frame.grid() # Unhide the posture
            self.posture_button.config(text="Hide Posture")

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
        top_view_label = tk.Label(self.views_frame, text="Top View", font=("Arial", 12))
        top_view_label.grid(column=0, row=0, sticky="nsew")

        front_view_label = tk.Label(self.views_frame, text="Front View", font=("Arial", 12))
        front_view_label.grid(column=1, row=0, sticky="nsew")

        side_view_label = tk.Label(self.views_frame, text="Side View", font=("Arial", 12))
        side_view_label.grid(column=0, row=2, sticky="nsew")  # Position adjusted for the layout

        plot_view_label = tk.Label(self.views_frame, text="3D Plot", font=("Arial", 12))
        plot_view_label.grid(column=1, row=2, sticky="nsew")  # Position adjusted for the layout

        # Creating and embedding matplotlib figures for top, front, side, and additional 3D plots
        self.top_view_fig, self.top_view_ax = self.create_plot_for_view()
        self.embed_plot_in_frame(self.top_view_fig, 0, 1)

        self.front_view_fig, self.front_view_ax = self.create_plot_for_view()
        self.embed_plot_in_frame(self.front_view_fig, 1, 1)

        self.side_view_fig, self.side_view_ax = self.create_plot_for_view()
        self.embed_plot_in_frame(self.side_view_fig, 0, 3)  # Adjust row for layout

        self.additional_3d_fig, self.additional_3d_ax = self.create_plot_for_view()
        self.embed_plot_in_frame(self.additional_3d_fig, 1, 3)  # Adjust row for layout

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

        if self.latest_landmarks:
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
        self.cap=cv2.VideoCapture(0)
        # Set the resolution of the webcam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Displaying video feed
        self.video_label = tk.Label(self.root)
        self.video_label.grid(column=1, row=0, sticky="nsew")

    def toggle_plot(self):
        self.plotting_active = not self.plotting_active # Change from True to False
        self.toggle_button.config(text="Start" if not self.plotting_active else "Stop")        

    def create_3d_plot(self):
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(column=0, row=0, sticky="nsew")

    def update(self):
        success, image = self.cap.read()                                                #new update method

        if success:

            # Convert the BGR image to RGB, then Tkinter PhotoImage
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Calculate pose using MediaPipe on the captured frame
            results = self.pose.process(rgb_image)

            # If landmarks were detected, draw them on the image
            if results.pose_landmarks:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(rgb_image, results.pose_landmarks, connections=self.custom_connections, landmark_drawing_spec=self.custom_style)

#mp_drawing.draw_landmarks(image, results.pose_landmarks, connections = custom_connections, landmark_drawing_spec=custom_style)
            # Convert the RGB image to a Tkinter PhotoImage    
            im_pil = Image.fromarray(rgb_image)
            image_tk = ImageTk.PhotoImage(image=im_pil)

            # Update the video label with the new image
            self.video_label.configure(image=image_tk)
            self.video_label.image = image_tk 

            if results.pose_world_landmarks:
                self.latest_landmarks = results.pose_world_landmarks

        # Update 3D plot with the new pose landmarks (THIS IS WHERE THE PLOTS ARE UPDATING)
                if self.multiple_views_active == True and results.pose_world_landmarks:
                    self.update_views()

                if self.plotting_active == True and results.pose_world_landmarks:
                    self.update_3d_plot(results.pose_world_landmarks)
                    
                    if self.posture_active == True:
                        for idx, group in enumerate(LANDMARK_GROUPS):
                            csv_file = r'C:\RU\MASc\GIT\Pose_Model\angle_ranges.csv'
                            _, _, _, angD = get_coordinates(results.pose_world_landmarks, group)
                            idcx = get_data(csv_file, idx, angD)
                            if idcx is not None:
                                print(idx, idcx)
                                self.change_light_color(idx, idcx)
                else:
                    self.update_main_live_feed(image, results)

        self.root.after(10, self.update)

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
    # see a working GUI by next wednesday 