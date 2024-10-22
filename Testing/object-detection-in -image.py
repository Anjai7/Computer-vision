import cv2
import supervision as sv
from ultralytics import YOLOv10

model = YOLOv10(f'/home/anjai/Downloads/runs/detect/train5/weights/best.pt')
image = cv2.imread(f'jupyter/bus.jpg')
results = model(image) [0]
detections = sv.Detections.from_ultralytics(results)

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = bounding_box_annotator.annotate(
     scene=image, detections=detections)
annotated_image = label_annotator.annotate(
     scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)
