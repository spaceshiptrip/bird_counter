import argparse
import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def main(image_path):
    # 1. Load the pre-trained YOLOv5s model from Ultralytics
    #    This will download the model if it is not already available.
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
    # model = YOLO('yolov8n.pt')

    # Set the confidence and IoU thresholds as model attributes
    model.conf = 0.25  # Confidence threshold
    model.iou  = 0.2   # IoU threshold for NMS

    # 2. Perform inference on the image
    results = model(image_path)
    #results = model(image_path, conf_thres=0.25, iou_thres=0.4)
    #results = model(image_path, iou_thres=0.4)


    # 3. Parse results: the results are available as a pandas DataFrame.
    #    Each row corresponds to a detected object with its class, bounding box, etc.
    detections = results.pandas().xyxy[0]

    # 4. Filter the detections to only those labeled as 'bird'
    #    (COCO dataset uses the label 'bird' for birds.)
    bird_detections = detections[detections['name'] == 'bird']

    # 5. Count the number of bird detections
    num_birds = len(bird_detections)
    print(f"Number of birds detected: {num_birds}")

    # Optional: Draw the detections on the image and display them.
    # The 'results' object has a built-in method to show the image.
    results.show()  # This will open a window displaying the image with bounding boxes

    # Alternatively, you can save the resulting image with bounding boxes:
    results.save(save_dir='output')  # The image with detections is saved in the 'output' folder.

if __name__ == '__main__':
    # Set up argument parser to allow specifying the image file via command line
    parser = argparse.ArgumentParser(description='Detect and count birds in an image using YOLOv5.')
    parser.add_argument('image_path', type=str, help='Path to the image file')

    args = parser.parse_args()

    main(args.image_path)

