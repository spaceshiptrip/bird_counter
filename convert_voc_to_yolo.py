import os
import argparse
import xml.etree.ElementTree as ET

# Define class mapping (Modify as needed)
class_mapping = {"bird": 0}  # Add more classes if needed

def convert_voc_to_yolo(xml_file, output_dir=None):
    """ Convert Pascal VOC XML annotation to YOLO format. """

    # Parse XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get image filename and size
    image_filename = root.find("filename").text
    image_path = root.find("path").text
    image_width = int(root.find("size/width").text)
    image_height = int(root.find("size/height").text)

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(xml_file)  # Save in same directory as XML

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define output file path
    txt_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(xml_file))[0] + ".txt")

    with open(txt_filename, "w") as f:
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in class_mapping:
                continue  # Skip unknown classes

            class_id = class_mapping[class_name]
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            # Normalize YOLO format
            x_center = ((xmin + xmax) / 2) / image_width
            y_center = ((ymin + ymax) / 2) / image_height
            bbox_width = (xmax - xmin) / image_width
            bbox_height = (ymax - ymin) / image_height

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

    print(f"Converted: {xml_file} → {txt_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XML to YOLO format")
    parser.add_argument("xml_file", help="Path to the XML annotation file")
    parser.add_argument("--output_dir", help="Optional: Output directory for YOLO txt file", default=None)

    args = parser.parse_args()
    convert_voc_to_yolo(args.xml_file, args.output_dir)

