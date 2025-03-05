# Install
#pip install ultralytics

# Import
from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n.pt')  # load the nano model (smallest)
# model = YOLO('yolov8s.pt')  # small model
# model = YOLO('yolov8m.pt')  # medium model
# model = YOLO('yolov8l.pt')  # large model
# model = YOLO('yolov8x.pt')  # extra-large model
model = YOLO(r'C:\Users\Tanarat\Desktop\IMP JOB\IMP PROJECT\Leak Detect\runs\detect\train3\weights\best.pt')
# Make predictions
results = model(r'C:\Users\Tanarat\Desktop\IMP JOB\IMP PROJECT\Leak Detect\pic\img1.jpg')