import os
import cv2
import supervision as sv
from ultralytics import YOLO
 
model = YOLO(f'/home/anjai/Downloads/yolo11x.pt')
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to read camera feed")
    exit()

# Directory to save images
img_counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Print the tensors
    print(results.boxes.xyxy)
    print(results.boxes.conf)
    print(results.boxes.cls)

    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    # Display the webcam feed
    cv2.imshow('Webcam', annotated_image)

    # Wait for a key press
    k = cv2.waitKey(1)

    if k % 256 == 27:  # Press 'Esc' to exit
        print("Escape hit, closing...")
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()


