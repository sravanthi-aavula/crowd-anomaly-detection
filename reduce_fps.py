import cv2
import os

# Input and output paths
input_path = 'input/1338598-hd_1920_1080_30fps.mp4'
output_path = 'input/1338598-hd_1920_1080_30fps_lowfps.mp4'

# Set desired FPS
desired_fps = 5

# Open the video
cap = cv2.VideoCapture(input_path)

# Get original video details
orig_fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, desired_fps, (width, height))

frame_count = 0
frame_skip = int(orig_fps // desired_fps)

print(f"Original FPS: {orig_fps} --> New FPS: {desired_fps}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip == 0:
        out.write(frame)

    frame_count += 1

cap.release()
out.release()
print("Done! Video saved as", output_path)
