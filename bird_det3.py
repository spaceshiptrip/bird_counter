#!/usr/bin/env python3
from ultralytics import YOLO
import argparse
import pandas as pd
import os

def main(image_path):
    # Load the YOLOv8 model (using the nano version as an example)
    model = YOLO('yolov8x.pt')

    # Run inference on the image.
    results = model(image_path)

    # Since we're processing one image, extract the first result.
    result = results[0]

    # Convert the detection boxes into a pandas DataFrame.
    # result.boxes.data tensor has shape [N, 6]: [xmin, ymin, xmax, ymax, confidence, class]
    boxes = result.boxes
    data = boxes.data.cpu().numpy()  # Convert tensor to numpy array
    df = pd.DataFrame(data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])

    # Map the numeric class values to their corresponding names using result.names.
    df['name'] = df['class'].apply(lambda c: result.names[int(c)])

    # Filter the detections for those labeled as 'bird'
    bird_df = df[df['name'] == 'bird']

    # Count the number of bird detections
    num_birds = len(bird_df)
    print(f"Number of birds detected: {num_birds}")

    # Optionally display the annotated image
    result.show()

    # Save the annotated image.
    # Ensure the output directory exists.
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Provide a filename with a valid image extension, e.g., "result.jpg"
    output_filename = os.path.join(output_dir, "result.jpg")
    result.save(output_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect and count birds using YOLOv8.')
    parser.add_argument('image_path', type=str, help='Path to the image file')

    args = parser.parse_args()
    main(args.image_path)

