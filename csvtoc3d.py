import csv
import os
import numpy as np
import kineticstoolkit.lab as ktk


# Usage
csv_file = 'sitting_t-pose.csv'
c3d_file = 'sitting_t-pose.c3d'

# Read CSV data
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)  # Skip header row
    data = np.array([row for row in reader], dtype=float)

# Extract time column
time = data[:, 0]

# Extract x, y, z values
num_joints = (data.shape[1] - 1) // 3
x_values = data[:, 1:num_joints + 1]
y_values = data[:, num_joints + 1:2*num_joints + 1]
z_values = data[:, 2*num_joints + 1:]

points = ktk.TimeSeries()
points.time = time
#points.data[headers[1]] = (INSERT HERE) meow meow meow :3
for i in range(1, min(30, len(headers))): # Change the 30 to the number of headers you have in your current file] (i.e. LASI, LPSI...)
    points.data[headers[i]] = np.column_stack([x_values[:, i], y_values[:,i], z_values[:,i], np.ones(len(time))])

x = points.data[headers[1]]
ktk.write_c3d(c3d_file, points=points)
