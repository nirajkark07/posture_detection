import csv
import matplotlib.pyplot as plt
import math

def calculate_angle(j1, j2, j3): # Shoulder, elbow, wrist
    """_summary_
    Args:
        j1, j2, j3: Three joints.
    Output:
        angD: Interior angle bounded by three joints <j1j2j3
    """
    vector_j2_j1 = (j1.x - j2.x, j1.y - j2.y, j1.z - j2.z)
    vector_j2_j3 = (j3.x - j2.x, j3.y - j2.y, j3.z - j2.z)
    # dot_product = vector_j2_j1[0]*vector_j2_j3[0] + vector_j2_j1[1]*vector_j2_j3[1] + vector_j2_j1[2]*vector_j2_j3[2]
    magnitude_j2_j1 = math.sqrt((vector_j2_j1[0] ** 2) + (vector_j2_j1[1] ** 2)+ (vector_j2_j1[2] ** 2))
    norm_j2_j1 = (vector_j2_j1[0]/magnitude_j2_j1, vector_j2_j1[1]/magnitude_j2_j1, vector_j2_j1[2]/magnitude_j2_j1)
    magnitude_j2_j3 = math.sqrt((vector_j2_j3[0] ** 2) + (vector_j2_j3[1] ** 2)+ (vector_j2_j3[2] ** 2))
    norm_j2_j3 = (vector_j2_j3[0]/magnitude_j2_j3, vector_j2_j3[1]/magnitude_j2_j3, vector_j2_j3[2]/magnitude_j2_j3)

    dot_product = norm_j2_j1[0]*norm_j2_j3[0] + norm_j2_j1[1]*norm_j2_j3[1] + norm_j2_j1[2]*norm_j2_j3[2]

    angR = math.acos(dot_product)
    angD = round(math.degrees(angR))

    return angD

def get_color(csv_file, idx, angD):
    """_summary_
    Args:
        csv_file: link to the csv fil
        idx: index of interested measurement
        angD: angle of joints
    Output:
        color: color the label the instrument
    """
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if int(row['idx']) == idx and int(row['start_range']) <= angD <= int(row['end_range']):
                return row['color']
    return 'g' # Return None if no matching range is found

def get_coordinates(landmarks, group):
    """_summary_
    Args:
        landmarks: Detections
        group: Landmark group
    Output:
        plotX, plotY, plotZ: Points to plot
    """
    plotX, plotY, plotZ = [], [], []
    if len(group) == 3:
        j1 = landmarks.landmark[group[0]]
        j2 = landmarks.landmark[group[1]]
        j3 = landmarks.landmark[group[2]]
        plotX = [j1.x, j2.x, j3.x]
        plotY = [j1.y, j2.y, j3.y]
        plotZ = [j1.z, j2.z, j3.z]
        angD = calculate_angle(j1, j2, j3)
        
    if len(group) == 2:
        j1 = landmarks.landmark[group[0]]
        j2 = landmarks.landmark[group[1]]
        plotX = [j1.x, j2.x]
        plotY = [j1.y, j2.y]
        plotZ = [j1.z, j2.z]
        angD = 0

    return plotX, plotY, plotZ, angD
