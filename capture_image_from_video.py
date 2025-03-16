import cv2
import os


def capture_image(video_path, save_image_folder):
    os.makedirs(save_image_folder, exist_ok=True)

    vid = cv2.VideoCapture(video_path)

    frame_id = 0
    while True:
        ret, frame = vid.read()

        if not ret:
            break

        cv2.imwrite(os.path.join(save_image_folder,
                    f"frame_{frame_id}.jpg"), frame)
        frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_image(video_path="fall.mp4", save_image_folder="image_from_video")
