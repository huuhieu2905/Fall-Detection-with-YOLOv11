# 1.Install code
Run this command in Terminal / Command Prompt
```bash
git clone https://github.com/huuhieu2905/Fall-Detection-with-YOLOv11.git
cd Fall-Detection-with-YOLOv11
```

# 2.Set up environment
## Set up conda environment
```bash
conda create -n yolo python==3.10 -y
conda activate yolo
```
## Install requirements
Run this command
```bash
pip install -r requirements.txt
```

# 3.Demo
## 1. Run demo with bounding box algorithm
```bash
python demo_video_bbox_algorithm.py
```

## 2. Run demo with keypoints algorithm
```bash
python demo_video_keypoints_algorithm.py
```

## Note
<p>We will update README file after we update full code.</p>

