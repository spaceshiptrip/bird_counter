import os
import xml.etree.ElementTree as ET

# Define paths
xml_dir = "annotations_xml/"  # Folder containing XML files
output_dir = "labels_yolo/"   # Folder to save YOLO annotations
image_width, image_height = 3024, 4032  # Replace with your actual image size

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define class mapping
class_mapping = {"bird": 0}  # Add more classes if needed

# Function to convert VOC to YOLO format
def convert_voc_to_yolo(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_filename = root.find("filename").text
    txt_filename = os.path.join(output_dir, image_filename.replace(".jpg", ".txt"))

    with open(txt_filename, "w") as f:
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in class_mapping:
                continue

            class_id = class_mapping[class_name]
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            # Normalize coordinates
            x_center = ((xmin + xmax) / 2) / image_width
            y_center = ((ymin + ymax) / 2) / image_height
            bbox_width = (xmax - xmin) / image_width
            bbox_height = (ymax - ymin) / image_height

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

# Convert all XML files in the directory
for xml_file in os.listdir(xml_dir):
    if xml_file.endswith(".xml"):
        convert_voc_to_yolo(os.path.join(xml_dir, xml_file))

print("Conversion completed! YOLO annotations saved in", output_dir)

