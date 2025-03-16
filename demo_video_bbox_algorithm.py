import cv2
import cvzone
import math
from ultralytics import YOLO
import os


def demo_video(video_url, save_results, checkpoint_path, stream=True):
    os.makedirs(save_results, exist_ok=True)

    if stream:
        cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(video_url)

    model = YOLO(checkpoint_path)
    cap.set(cv2.CAP_PROP_FPS, 15)
    classnames = []
    with open('classes.txt', 'r') as f:
        classnames = f.read().splitlines()

    frame_id = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)

        for info in results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = box.conf[0]
                class_detect = box.cls[0]
                class_detect = int(class_detect)
                class_detect = classnames[class_detect]
                conf = math.ceil(confidence * 100)

                # implement fall detection using the coordinates x1,y1,x2
                height = y2 - y1
                width = x2 - x1
                threshold = height - width

                if conf > 40 and class_detect == 'person':
                    cvzone.cornerRect(
                        frame, [x1, y1, width, height], l=30, rt=6)
                    cvzone.putTextRect(frame, f'{class_detect}', [
                        x1 + 8, y1 - 12], thickness=2, scale=2)

                    if threshold < 0:
                        cvzone.putTextRect(frame, 'Fall Detected', [
                            height, width], thickness=2, scale=2)
                        cv2.imwrite(os.path.join(
                            save_results, f"frame_{frame_id}_fall_detected.jpg"), frame)

                    else:
                        pass
        frame_id += 1

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # demo_video(video_url="rtsp://admin:secam123@192.168.68.22:554/Streaming/Channels/102?transportmode=unicast&profile=Profile_2",
    #            save_results="frame_fall_detect",
    #            checkpoint_path="yolo11m.pt")
    demo_video(video_url="istockphoto-1446309379-640_adpp_is.mp4",
               save_results="frame_fall_detect",
               checkpoint_path="yolo11m.pt",
               stream=False)
