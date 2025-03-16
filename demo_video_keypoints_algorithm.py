import cv2
import cvzone
import math
from ultralytics import YOLO
import os
import numpy as np


# Thresholds
CRITICAL_VELOCITY = 0.009  # m/s
CRITICAL_ANGLE = 45  # degrees
WIDTH_HEIGHT_RATIO_THRESHOLD = 1.0
FRAME_INTERVAL = 1  # Process every 5 frames
FPS = 30  # Frames per second
TIME_DIFF = FRAME_INTERVAL / FPS  # Time difference between frames

# Function to calculate velocity of hip joint


def calculate_velocity(prev_hip, curr_hip, time_diff):
    if prev_hip is None or curr_hip is None:
        return 0
    velocity = abs(curr_hip[1] - prev_hip[1]) / time_diff
    return velocity

# Function to calculate angle of body using head and hips


def calculate_angle(head, hips):
    if head is None or hips is None:
        return 90  # Assume standing if no detection
    delta_x = abs(head[0] - hips[0])
    delta_y = abs(head[1] - hips[1])
    angle = np.arctan2(delta_y, delta_x) * 180 / np.pi
    return angle

# Function to calculate width-to-height ratio of bounding box


def calculate_width_height_ratio(bbox):
    if bbox is None:
        return 0
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width / height

# Function to calculate center hip keypoint


def calculate_center(keypoints):
    if keypoints is None:
        return None
    left_knee = keypoints[13]  # Left knee
    left_ankle = keypoints[15]  # Left ankle
    center = ((left_knee[0] + left_ankle[0]) / 2,
              (left_knee[1] + left_ankle[1]) / 2)
    return center

# Function to get head keypoint


def get_head_keypoint(keypoints):
    if keypoints is None:
        return None
    return keypoints[0]  # Head keypoint


# Function to draw bounding box
def draw_bounding_box(frame, bbox, color):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2),
                  color, 2)  # Green bounding box


def draw_keypoints(frame, keypoints, color):
    for keypoint in keypoints:
        cv2.circle(frame, (int(keypoint[0]), int(
            keypoint[1])), radius=1, color=color, thickness=2, lineType=cv2.LINE_AA)


def demo_video(video_url, save_results, save_video, checkpoint_pose_estimation, stream):

    os.makedirs(save_results, exist_ok=True)

    if stream:
        cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(video_url)

    ret, frame = cap.read()
    height, width, _ = frame.shape
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30

    out = cv2.VideoWriter(save_video, fourcc, fps, frame_size)

    model = YOLO(checkpoint_pose_estimation)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    prev_hip = None
    frame_count = 0
    flag = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # if flag == 1:
        #     out.write(frame)

        if frame_count % FRAME_INTERVAL != 0:
            continue

        results = model(frame)

        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints.xy.cpu().numpy()
            for box, keypoints in zip(boxes, keypoints):
                if box.cls[0] == 0:
                    bbox = box.xyxy[0]
                    draw_bounding_box(frame, bbox, (0, 255, 0))
                    draw_keypoints(frame, keypoints, (0, 255, 255))
                    hip = calculate_center(
                        keypoints) if keypoints is not None else None
                    head = get_head_keypoint(
                        keypoints) if keypoints is not None else None
                    ratio = calculate_width_height_ratio(bbox)

                    if hip is not None and head is not None:
                        velocity = calculate_velocity(prev_hip, hip, TIME_DIFF)
                        angle = calculate_angle(head, hip)

                        if velocity > CRITICAL_VELOCITY and angle < CRITICAL_ANGLE and ratio > WIDTH_HEIGHT_RATIO_THRESHOLD:
                            print("Velocity", velocity)
                            print("Angle", angle)
                            print("Ratio", ratio)
                            cv2.putText(frame, 'FALL DETECTED!', (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            draw_bounding_box(frame, bbox, (0, 0, 255))
                            out.write(frame)
                            flag = 1
                            if stream is True:
                                cv2.imwrite(os.path.join(
                                    save_results, f"frame_{frame_count}.jpg"), frame)
                            else:
                                cv2.imwrite(os.path.join(
                                    save_results, f"{os.path.basename(video_url)}_{frame_count}.jpg"), frame)
                        else:
                            flag = 0
                        prev_hip = hip

        cv2.imshow("Fall Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo_video("istockphoto-1446309379-640_adpp_is.mp4", save_results="results_pts", save_video="out_1.mp4",
               checkpoint_pose_estimation="yolo11m-pose.pt", stream=False)
    # demo_video("rtsp://admin:secam123@192.168.68.22:554/Streaming/Channels/102?transportmode=unicast&profile=Profile_2", save_results="results_pts",
    #            checkpoint_pose_estimation="yolo11m-pose.pt", stream=True)
