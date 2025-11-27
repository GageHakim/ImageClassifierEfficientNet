
import json
import os
from PIL import Image

def extract_and_label_bacteria(json_path):
    """
    Extracts bacteria from an image based on a JSON file, labels them,
    and saves them to the 'bacteriaLabeled' directory.

    Args:
        json_path (str): The path to the JSON file.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error reading JSON file: {json_path}")
        return

    labels = data.get("labels", [])
    if not labels:
        return

    base_path, _ = os.path.splitext(json_path)
    image_path = base_path + ".jpg"

    if not os.path.exists(image_path):
        return

    try:
        image = Image.open(image_path)
    except IOError:
        print(f"Error opening image file: {image_path}")
        return

    output_dir = "bacteriaLabeled"
    base_filename = os.path.splitext(os.path.basename(json_path))[0]

    for label in labels:
        bacteria_class = label.get("class")
        if not bacteria_class:
            continue

        x = label.get("x")
        y = label.get("y")
        width = label.get("width")
        height = label.get("height")

        if any(v is None for v in [x, y, width, height]):
            continue

        # Create a unique filename for the cropped image
        bacterium_id = label.get("id", "")
        cropped_filename = f"{base_filename}_{bacterium_id}_{bacteria_class}.jpg"
        cropped_filepath = os.path.join(output_dir, cropped_filename)

        # Crop the image
        box = (x, y, x + width, y + height)
        cropped_image = image.crop(box)

        # Save the cropped image
        cropped_image.save(cropped_filepath)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        json_filepath = sys.argv[1]
        extract_and_label_bacteria(json_filepath)
    else:
        # If no arguments are provided, find all json files and process them
        for root, _, files in os.walk("higher-resolution"):
            for file in files:
                if file.endswith(".json"):
                    json_path = os.path.join(root, file)
                    extract_and_label_bacteria(json_path)
