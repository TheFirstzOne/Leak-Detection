import cv2
from ultralytics import YOLO

# Load the model
model = YOLO('runs/detect/train3/weights/best.pt')

# Open webcam (or video file)
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or file path for video

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Leak Detection", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()