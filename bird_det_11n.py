#!/usr/bin/env python3
from ultralytics import YOLO
import argparse
import pandas as pd
import os

def main(image_path):
    # Load a pretrained YOLO model.
    # For YOLOv11, you can choose variants such as "yolov11n.pt", "yolov11s.pt", etc.
    model = YOLO("yolo11x.pt")

    # Run inference on the image.
    results = model(image_path)

    # For a single image, get the first result.
    result = results[0]

    # (Optional) Convert detection boxes to a pandas DataFrame.
    # The boxes data is a tensor with columns: [xmin, ymin, xmax, ymax, confidence, class]
    boxes_tensor = result.boxes.data.cpu().numpy()  # Convert tensor to numpy array
    df = pd.DataFrame(boxes_tensor, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])

    # Map the numeric class values to names using result.names.
    df['name'] = df['class'].apply(lambda c: result.names[int(c)])

    # (Optional) Filter the DataFrame for a specific class (e.g., 'bird')
    bird_df = df[df['name'] == 'bird']
    num_birds = len(bird_df)
    print(f"Number of birds detected: {num_birds}")

    # Display the annotated image.
    result.show()

    # Save the annotated image.
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "result.jpg")
    result.save(output_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect objects using Ultralytics YOLO (v11 API)")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()
    main(args.image_path)

