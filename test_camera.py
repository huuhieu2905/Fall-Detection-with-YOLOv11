import cv2
import os
import vlc
import numpy as np
import time


def load_video(video_url, folder_save_frame):
    os.makedirs(folder_save_frame, exist_ok=True)
    cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Can't read video")
        exit()

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Connection timed out")
            cap.release()
            cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 10)
            continue

        print("Frame id:", frame_id)
        cv2.imwrite(os.path.join(folder_save_frame,
                    f"frame_{frame_id}.jpg"), frame)
        frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def load_video_vlc(video_url):
    instance = vlc.Instance("--verbose=2")
    player = instance.media_player_new()
    media = instance.media_new(video_url)

    media.add_option("logfile=vlc.log")

    player.set_media(media)

    # Start play
    player.play()
    time.sleep(2)  # Wait time

    frame_id = 0
    while True:
        frame = player.video_take_snapshot(
            0, f"frame_camera/frame_{frame_id}.jpg", 0, 0)
        # if frame:
        #     img = np.array(frame[0])  # Convert frame to numpy array
        #     cv2.imwrite(f"frame_camera/frame_{frame_id}.jpg", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_id += 1

    player.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    load_video(video_url="rtsp://admin:secam123@192.168.68.22:554/Streaming/Channels/102?transportmode=unicast&profile=Profile_2",
               folder_save_frame='frame_camera_opencv')
    # load_video_vlc(video_url="rtsp://admin:secam123@192.168.68.22:554/Streaming/Channels/102?transportmode=unicast&profile=Profile_2")
