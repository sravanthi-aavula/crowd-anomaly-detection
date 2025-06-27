import cv2
import numpy as np
from sort.sort import Sort
import time
import math
import json
import os
import pygame
from collections import deque


pygame.init()
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("input/notification-alert-269289.mp3")
alert_triggered = False
# --- Load YOLOv4-tiny model ---
net = cv2.dnn.readNetFromDarknet("frontend/model/yolov4-tiny.cfg", "frontend/model/yolov4-tiny.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# --- Video Input ---
cap = cv2.VideoCapture("input/1338598-hd_1920_1080_30fps_lowfps.mp4")
tracker = Sort()

# --- Parameters ---
prev_centroids = {}
track_history = {}
anomalies = []
frame_count = 0

MAX_HISTORY = 15
IDLE_THRESHOLD = 5
IDLE_FRAMES = 10
SPEED_THRESHOLD = 25
ANGLE_CHANGE_THRESHOLD = 45
PROXIMITY_THRESHOLD = 50

# --- Utils ---
def get_centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def calculate_angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# --- Output setup ---
os.makedirs("output", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output/tracked_output.mp4", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)

    boxes, confidences = [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id == 0 and scores[class_id] > 0.4:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(scores[class_id]))

    detections = []
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        detections.append([x, y, x + w, y + h, confidences[i]])

    tracks = tracker.update(np.array(detections))

    ids = []
    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        ids.append(track_id)
        centroid = get_centroid((x1, y1, x2, y2))

        # Track centroid history
        if track_id not in track_history:
            track_history[track_id] = deque(maxlen=MAX_HISTORY)
        track_history[track_id].append(centroid)

        # Speed Anomaly
        if track_id in prev_centroids:
            old_centroid = prev_centroids[track_id]
            speed = euclidean(old_centroid, centroid)
            if speed > SPEED_THRESHOLD:
                anomalies.append({
                    "type": "speed",
                    "frame": int(frame_count),
                    "id": int(track_id),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
                cv2.putText(frame, "SPEED", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

        # Idle Anomaly
        if len(track_history[track_id]) >= IDLE_FRAMES:
            dist = euclidean(track_history[track_id][0], centroid)
            if dist < IDLE_THRESHOLD:
                anomalies.append({
                    "type": "idle",
                    "frame": int(frame_count),
                    "id": int(track_id),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
                cv2.putText(frame, "IDLE", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        # Turn Anomaly
        if len(track_history[track_id]) >= 3:
            mid_idx = len(track_history[track_id]) // 2
            angle = calculate_angle(track_history[track_id][0], track_history[track_id][mid_idx], centroid)
            if angle > ANGLE_CHANGE_THRESHOLD:
                anomalies.append({
                    "type": "sudden_turn",
                    "frame": int(frame_count),
                    "id": int(track_id),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
                cv2.putText(frame, "TURN", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)

        prev_centroids[track_id] = centroid

    # Proximity anomaly
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id1, id2 = ids[i], ids[j]
            if id1 in prev_centroids and id2 in prev_centroids:
                dist = euclidean(prev_centroids[id1], prev_centroids[id2])
                if dist < PROXIMITY_THRESHOLD:
                    anomalies.append({
                        "type": "close_proximity",
                        "frame": int(frame_count),
                        "id_pair": [int(id1), int(id2)],
                        "distance": float(round(dist, 2))
                    })

    out.write(frame)
    cv2.imshow("Anomaly Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Save JSON output
for anomaly in anomalies:
    for key in anomaly:
        if isinstance(anomaly[key], (np.int32, np.int64, np.float32, np.float64)):
            anomaly[key] = int(anomaly[key]) if isinstance(anomaly[key], (np.int32, np.int64)) else float(anomaly[key])

with open("output/anomalies.json", "w") as f:
    json.dump(anomalies, f, indent=2)

print("✅ Tracking complete. Outputs saved to output/")  