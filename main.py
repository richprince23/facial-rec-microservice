from PIL import Image
import cv2
import numpy as np

def convert_to_rgb(image_path):
    try:
        with Image.open(image_path) as img:
            # Convert to RGB
            rgb_img = img.convert('RGB')
            # Convert to 8-bit depth (optional, depending on your needs)
            if rgb_img.mode != 'RGB':
                rgb_img = rgb_img.convert('RGB')  # Ensure RGB mode if not already
            rgb_img = np.array(rgb_img)
            return rgb_img
    except IOError as e:
        print(f"Error opening image: {e}")
        return None
    

def check_image_format_and_depth(image_path):
    try:
        # Open the image with Pillow to detect mode
        image = Image.open(image_path)
        mode = image.mode
        print(f"Image mode: {mode}")  # Debugging print

        # Get the bit depth
        if hasattr(image, 'bits'):
            bit_depth = image.bits
        elif mode in ['RGB', 'RGBA']:
            bit_depth = 8 * len(mode)  # 24 for RGB, 32 for RGBA
        elif mode == 'L':
            bit_depth = 8  # 8-bit for grayscale
        else:
            bit_depth = 'Unknown'
        print(f"Bit depth: {bit_depth}")  # Debugging print

        # If the image is in RGB or RGBA mode, we can use OpenCV to check BGR/RGB
        if mode in ["RGB", "RGBA"]:
            # Convert the image to a numpy array
            image_np = np.array(image)
            print(f"Image shape: {image_np.shape}")  # Debugging print
            
            # Check the format by examining the first pixel
            first_pixel = image_np[0, 0]
            print(f"First pixel: {first_pixel}")  # Debugging print
            
            if len(first_pixel) >= 3 and first_pixel[0] > first_pixel[2]:  # Blue value is higher than Red value
                color_format = "BGR"
            else:
                color_format = "RGB"
            return f'{mode} ({color_format}), {bit_depth}-bit'
        else:
            return f'{mode}, {bit_depth}-bit'
    except Exception as e:
        print(f"An error occurred: {e}")  # Debugging print
        return None

# Example usage
image_path = 'imgs/3116.png'
file = convert_to_rgb(image_path)
format_depth = check_image_format_and_depth(file)
print(f'The image format and depth is {format_depth}')