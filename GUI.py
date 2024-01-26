import tkinter as tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.plot_pose_live import plot_world_landmarks
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

        # Create 3D plot on the left side
        self.create_3d_plot()

        # Setup mediapipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_tracking_confidence=0.5,
            min_detection_confidence=0.5,
            model_complexity=1,
            smooth_landmarks=True,
        )

        # Open webcam
        self.cap = cv2.VideoCapture(0)  # Change index if needed

        # Start main loop
        self.root.after(0, self.update)
        self.root.mainloop()

    def create_3d_plot(self):
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111, projection='3d')

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    def update(self):
        success, image = self.cap.read()

        if success:
            # Calculate pose
            results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Update 3D plot
            self.update_3d_plot(results.pose_world_landmarks)

            # Draw image
            cv2.imshow("MediaPipePose", cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            self.cap.release()
            cv2.destroyAllWindows()
            self.root.destroy()
        else:
            self.root.after(10, self.update)

    def update_3d_plot(self, landmarks):
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