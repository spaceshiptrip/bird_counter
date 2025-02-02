#!/usr/bin/env python3
from ultralytics import YOLO
import argparse
import pandas as pd

def main(image_path):
    # Load the YOLOv8 model (using the nano version as an example).
    # Other variants include 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
    model = YOLO('yolov8n.pt')

    # Run inference on the image.
    # YOLOv8 returns a list of Results objects (one per image)
    results = model(image_path)

    # Since we're processing one image, extract the first result.
    result = results[0]

    # Convert the detection boxes into a pandas DataFrame.
    # The result.boxes.data tensor typically has shape [N, 6]:
    # columns: [xmin, ymin, xmax, ymax, confidence, class]
    boxes = result.boxes
    data = boxes.data.cpu().numpy()  # Convert tensor to numpy array
    df = pd.DataFrame(data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])

    # Map the numeric class values to their corresponding names using result.names.
    # result.names is typically a dictionary mapping class IDs to class names.
    df['name'] = df['class'].apply(lambda c: result.names[int(c)])

    # Filter the detections for those labeled as 'bird'
    bird_df = df[df['name'] == 'bird']

    # Count the number of bird detections
    num_birds = len(bird_df)
    print(f"Number of birds detected: {num_birds}")

    # Optionally display the annotated image
    result.show()

    # Optionally save the annotated image in an output folder
    result.save(save_dir='output')

if __name__ == '__main__':
    # Set up argument parser to allow specifying the image file via the command line
    parser = argparse.ArgumentParser(description='Detect and count birds using YOLOv8.')
    parser.add_argument('image_path', type=str, help='Path to the image file')

    args = parser.parse_args()
    main(args.image_path)

