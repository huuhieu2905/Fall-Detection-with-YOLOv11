from ultralytics import YOLO
import cv2
import os


def demo_image(image_path, save_results_folder):
    os.makedirs(save_results_folder, exist_ok=True)
    img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)

    # Load pretrained YOLO11n model
    model = YOLO("yolo11m-pose.pt")

    # Run inference on the source
    results = model(image_path)

    for r in results:
        idx = 0
        for pts, conf in zip(r.keypoints.xy.numpy()[0], r.keypoints.conf.numpy()[0]):
            print(pts, conf)
            cv2.circle(img, (int(pts[0]), int(pts[1])), radius=1, color=(
                255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            # cv2.imwrite(os.path.join(save_results_folder,
            #                          f"{image_path.split('/')[-1].split('.')[0]}_{idx}.jpg"), img)
            idx += 1


if __name__ == "__main__":
    demo_image("image_test_1.jpg", "results")
