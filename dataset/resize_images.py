import os
from PIL import Image

def resize_and_pad_upscale(image_path, output_path, size=(224, 224), background_color=(0, 0, 0)):
    """
    Resizes an image to make its longest side match the target size,
    maintaining aspect ratio. It scales images both up and down.
    Pads the remaining space to make it a square.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the resized image.
        size (tuple): The target size (width, height).
        background_color (tuple): The RGB color for the padding.
    """
    try:
        img = Image.open(image_path)
        
        # Calculate the scaling factor
        if img.width > img.height:
            scale = size[0] / img.width
        else:
            scale = size[1] / img.height
            
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        
        # Resize the image using the calculated factor
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        new_img = Image.new("RGB", size, background_color)
        
        # Calculate position to paste the image so it's centered
        paste_x = (size[0] - new_width) // 2
        paste_y = (size[1] - new_height) // 2
        
        new_img.paste(img_resized, (paste_x, paste_y))
        new_img.save(output_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    input_dir = "bacteriaLabeled"
    output_dir = "bacteria_resized_224"
    target_size = (224, 224)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            resize_and_pad_upscale(input_path, output_path, size=target_size)