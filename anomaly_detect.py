import math

track_history = {}  # Stores recent positions for each ID

SPEED_THRESHOLD = 25     # Tune based on resolution
LOITERING_THRESHOLD = 70 # Distance moved over last N frames
HISTORY_LENGTH = 10      # Number of frames to store per ID


def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def update_and_check_anomalies(detections):
    anomalies = []

    for det in detections:
        x1, y1, x2, y2, track_id = det
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        if track_id not in track_history:
            track_history[track_id] = []

        track_history[track_id].append((cx, cy))

        # Limit history length
        if len(track_history[track_id]) > HISTORY_LENGTH:
            track_history[track_id] = track_history[track_id][-HISTORY_LENGTH:]

        # Speed-based anomaly
        if len(track_history[track_id]) >= 2:
            speed = euclidean(track_history[track_id][-1], track_history[track_id][-2])
            if speed > SPEED_THRESHOLD:
                anomalies.append({
                    'id': track_id,
                    'type': 'sudden_movement',
                    'speed': speed
                })

        # Loitering-based anomaly
        if len(track_history[track_id]) == HISTORY_LENGTH:
            total_dist = sum(
                euclidean(track_history[track_id][i], track_history[track_id][i + 1])
                for i in range(HISTORY_LENGTH - 1)
            )
            if total_dist < LOITERING_THRESHOLD:
                anomalies.append({
                    'id': track_id,
                    'type': 'loitering',
                    'distance': total_dist
                })

    return anomalies