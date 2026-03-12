"""Motion Detection using OpenCV

This module opens the default webcam (or specified camera index) and detects
motion by comparing consecutive frames. It draws bounding boxes around regions
of movement, prints/logs timestamps, and handles empty-frame errors gracefully.

Usage:
    python motion_detection.py

Controls (when the video window is active):
    q or s : stop the camera and exit

The function `run_motion_detection(output_log="motion_times.txt",
camera_index=0)` can be imported by other scripts or the Streamlit dashboard.
"""

import cv2
import numpy as np
from datetime import datetime


def run_motion_detection(output_log="motion_times.txt", camera_index=0):
    cap = cv2.VideoCapture(camera_index)  # default camera
    if not cap.isOpened():
        print("Could not open video device")
        return

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    if frame1 is None or frame2 is None:
        print("Error: could not read initial frames from camera")
        cap.release()
        return

    motion_log = []
    frames = 0
    events = 0

    print("Press 'q' or 's' in the video window to stop")
    while True:
        # compute difference only if frames are valid
        if frame1 is None or frame2 is None:
            print("Warning: empty frame encountered, stopping")
            break

        diff = cv2.absdiff(frame1, frame2)
        if diff is None or diff.size == 0:
            print("Warning: diff image empty, skipping iteration")
            frame1 = frame2
            ret, frame2 = cap.read()
            if not ret:
                break
            continue

        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        motion = False
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            motion = True

        status_text = "Motion" if motion else "No Motion"
        cv2.putText(frame1, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        frames += 1
        if motion:
            events += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            motion_log.append(timestamp)
            print(f"Motion detected at {timestamp}")

        cv2.imshow("Motion Detector", frame1)

        frame1 = frame2
        ret, frame2 = cap.read()
        if not ret or frame2 is None:
            print("Frame grab failed, ending")
            break

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == ord('s'):
            print("Stop key pressed")
            break

    cap.release()
    cv2.destroyAllWindows()

    # save log
    with open(output_log, 'w') as f:
        for t in motion_log:
            f.write(t + "\n")

    accuracy = events / frames if frames else 0
    print(f"Processed {frames} frames, {events} motion events (accuracy={accuracy:.2%})")
    return {"frames": frames, "events": events, "accuracy": accuracy}


if __name__ == "__main__":
    run_motion_detection()
