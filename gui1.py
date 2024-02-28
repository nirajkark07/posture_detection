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
from mediapipe import solutions as mp
import mediapipe as mp
import cv2

LANDMARK_GROUPS = [
    [11, 13, 15],  # left arm
    [12, 14, 16],  # right arm
    [13, 11, 23],  # left elbow, left shoulder, left knee
    [14, 12, 24],  # right elbow, right shoulder, right torso
    [11, 23, 25],  # left shoulder, left hip, left knee
    [12, 24, 26],  # right shoulder, right hip, right knee
    [23, 25, 27],  # left hip, left knee, left ankle
    [24, 26, 28],  # right hip, right knee, right ankle
]

class PoseVisualizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pose Visualizer")
        self.latest_landmarks = None

        # Layout configuration to make video and plot side by side
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0,weight=1)

        # Create 3D plot on the left side
        self.create_3d_plot()

        #Initialize the live feed on the right side
        self.initialize_live_feed()

        # Create a label for displaying the video feed
        self.video_label = tk.Label(self.root)
        self.video_label.grid(column=1, row=0, sticky="nsew")

        self.root.bind('<Configure>', self.on_resize)

        # Initialize drawing ultility for MediaPipe pose landmarks
        self.mp_drawing = mp.solutions.drawing_utils

        # Add a control variable
        self.plotting_active = True

        # Create a start/stop button
        self.toggle_button = tk.Button(self.root, text="Stop", command=self.toggle_plot)
        self.toggle_button.grid(column=0, row=2, sticky="sw")  # Position the button at the bottom left

        # Create a View button
        self.views_button = tk.Button(self.root, text="Views", command=self.show_views)
        self.views_button.grid(column=0, row=4, sticky="sw")
        self.views_window = None

        # Setup mediapipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_tracking_confidence=0.5,
            min_detection_confidence=0.5,
            model_complexity=1,
            smooth_landmarks=True,
        )

        # Start main loop
        self.root.after(0, self.update)
        self.root.mainloop()

    def show_views(self):
        if self.views_window is not None:
            return
        self.views_window = tk.Toplevel(self.root)
        self.views_window.title("3D Graph Views")
        self.views_window.geometry("800x800")

        for i in range(2):
            self.views_window.columnconfigure(i, weight=1)
            self.views_window.rowconfigure(i, weight=1)

        self.top_view_fig, self.top_view_ax = self.create_plot()
        self.front_view_fig, self.front_view_ax = self.create_plot()
        self.side_view_fig, self.side_view_ax = self.create_plot()

        # Embedding matplotlib figures for top, front, and side views
        self.embed_plot(self.top_view_fig, 0, 0)
        self.embed_plot(self.front_view_fig, 1, 0)
        self.embed_plot(self.side_view_fig, 0, 1)

        # Simulate the live feed 
        self.prepare_live_feed_view()
        self.update_views()

    def create_plot(self):
        fig = Figure(figsize=(4, 4), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        return fig, ax

    def embed_plot(self, fig, col, row):
        canvas = FigureCanvasTkAgg(fig, master=self.views_window)
        canvas.draw()
        canvas.get_tk_widget().grid(column=col, row=row, sticky="nsew")
        
    
    def update_views(self):
        if self.views_window is None or not self.views_window.winfo_exists():
            return

        if self.latest_landmarks:
            self.update_single_view(self.top_view_ax, self.top_view_fig, 'top')
            self.update_single_view(self.front_view_ax, self.front_view_fig, 'front')
            self.update_single_view(self.side_view_ax, self.side_view_fig, 'side')
            self.update_live_feed()

        self.views_window.after(100, self.update_views)

    def update_single_view(self, ax, fig, view_type):
        ax.cla()  # Clear the current axes
        plot_world_landmarks(ax, self.latest_landmarks, LANDMARK_GROUPS)
        if view_type == "top":
            ax.view_init(elev=90, azim=-90)
        elif view_type == "front":
            ax.view_init(elev=0, azim=-90)
        elif view_type == "side":
            ax.view_init(elev=0, azim=0)
        fig.canvas.draw_idle()

    def on_resize(self, event):
        self.update_video_feed_size()
        pass

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
            im_pil = im_pil.resize((200, 150), Image.ANTIALIAS)
            image_tk = ImageTk.PhotoImage(image=im_pil)
            self.live_feed_label.configure(image=image_tk)
            self.live_feed_label.image = image_tk
        self.views_window.after(100, self.update_live_feed)        

    def initialize_live_feed(self):
        # Open webcam
        self.cap=cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Webcam not available.")
            return
        # Set the resolution of the webcam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Displaying video feed
        self.video_label = tk.Label(self.root)
        self.video_label.grid(column=1, row=0, sticky="nsew")

    def toggle_plot(self):
        self.plotting_active = not self.plotting_active
        self.toggle_button.config(text="Start" if not self.plotting_active else "Stop")        

    def create_3d_plot(self):
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(column=0, row=0, sticky="nsew")

    def update(self):
        success, image = self.cap.read()

        if success:
            # Calculate pose
            results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Update 3D plot
            self.latest_landmarks = results.pose_world_landmarks

            # Convert the BGR image to RGB, then Tkinter PhotoImage
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Calculate pose using MediaPipe on the captured frame
            results = self.pose.process(rgb_image)

            # If landmarks were detected, draw them on the image
            if results.pose_landmarks:
                # Draw landmarks
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(
                    rgb_image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,    
                )

            # Convert the RGB image to a Tkinter PhotoImage    
            im_pil = Image.fromarray(rgb_image)
            image_tk = ImageTk.PhotoImage(image=im_pil)

            # Update the video label with the new image
            self.video_label.configure(image=image_tk)
            self.video_label.image = image_tk 

            if results.pose_world_landmarks:
                self.latest_landmarks = results.pose_world_landmarks

            # Update 3D plot with the new pose landmarks
            if self.plotting_active and results.pose_world_landmarks:
                self.update_3d_plot(results.pose_world_landmarks)

            if self.views_window and self.views_window.winfo_exists():
                self.update_views()

        if cv2.waitKey(5) & 0xFF == 27:
            self.cap.release()
            cv2.destroyAllWindows()
            self.root.destroy()
        else:
            self.root.after(10, self.update)

    def on_views_window_close(self):
        self.views_window_close.destroy()
        self.views_window = None

    def update_3d_plot(self, landmarks):
        if self.plotting_active:

            self.ax.cla()

            # had to flip the z axis
            self.ax.set_xlim3d(-1, 1)
            self.ax.set_ylim3d(-1, 1)
            self.ax.set_zlim3d(1, -1)

            # get coordinates for each group and plot
            for idx, group in enumerate(LANDMARK_GROUPS):
                plot_world_landmarks(self.ax, landmarks)

            self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    gui = PoseVisualizerGUI(root)