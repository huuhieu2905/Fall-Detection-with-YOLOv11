from ultralytics import YOLO


model = YOLO("yolo11n-pose.pt")
results = model("000000004134.jpg")

count = 0
for r in results:
    print(r.boxes.xyxy)
    print(r.keypoints.xy)
