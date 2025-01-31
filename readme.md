
# Live 3D Reconstruction Demos

## Table of Contents

- [Description](#Description)
- [Installation](#Installation)
- [Running the Project](#running-the-project)
- [Demo](#Demo)

## Description
This project integrates:

    1. MediaPipe’s Posture Detection to capture human skeletal keypoints.
    2. Biomechanical Analysis to evaluate posture in real-time, providing feedback on user alignment and joint angles.

Potential use cases include:
    1. Physical therapy and fitness application
    2. Ergonomic assessments

## Installation
1. **Clone the repository**:
```bash
git https://github.com/nirajkark07/posture_detection.git
```
   
2. **Create Conda Enviornment**:

Create and activate conda enviornment.

```bash
conda create --name posture-env python=3.9
conda activate posture-env
```

Install Dependencies
```bash
pip install mediapipe opencv-python numpy
pip install torch torchvision
```

3. **Running the project**:

In the main repository folder run:
```bash
python App.py
```

## Demo
Below is a current demo of the project.


---


