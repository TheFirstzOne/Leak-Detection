from ultralytics import YOLO

# Start with a pretrained model (recommended for faster training and better results)
model = YOLO('yolov8n.pt')  # You can also try s, m, l, or x variants depending on your needs

# Train the model
results = model.train(
    data='C:/Users/Tanarat/Desktop/IMP JOB/IMP PROJECT/Leak Detect/dataset/data.yaml', 
    epochs=100,  # You might need more or fewer epochs depending on your dataset
    imgsz=640,   # Image size for training
    batch=16,    # Adjust based on your GPU memory
    patience=50  # Early stopping patience
)