import os
import shutil
import cv2
import glob

# Update these paths based on your actual folder structure:
image_directory = r"C:\Users\ragav\meteor_head_classifier\data1\pansy\cnn_images\cnn*.png"
sorted_image_base_path = r"C:\Users\ragav\meteor_head_classifier\data1\pansy\sorted_images"

def create_directories(base_path, categories):
    for category in categories:
        os.makedirs(os.path.join(base_path, category), exist_ok=True)

def move_image(image_path, category, base_path):
    dest_path = os.path.join(base_path, category, os.path.basename(image_path))
    shutil.move(image_path, dest_path)

def label_images(base_path):
    categories = {
        '1': 'head_echo',
        '2': 'head_echo_with_flare',
        '3': 'no_head_echo',
        '4': 'head_echo_and_unrelated_coh_scatter',
        '5': 'multiple_head_echoes',
        'x': 'exit',
        's': 'skip_image'
    }
    
    create_directories(base_path, categories.values())
    
    # Debug: check if the image directory exists and list its files
    image_folder = os.path.dirname(image_directory)
    print("Directory exists:", os.path.exists(image_folder))
    try:
        print("Files in directory:", os.listdir(image_folder))
    except Exception as e:
        print("Error listing files:", e)
    
    # Use glob to find images
    print("Using glob pattern:", image_directory)
    images = glob.glob(image_directory)
    print("Found images:", images)
    
    for image in images:
        image_path = image
        img = cv2.imread(image_path)
        if img is None:
            print("Failed to load image:", image_path)
            continue
        
        # Scale factor (increase by 4x)
        scale = 2.0
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        resized = cv2.resize(img, (width, height))
        
        cv2.imshow("Label Image", resized)
        print(f"Label the image {image} by pressing one of the keys:")
        for cat, desc in categories.items():
            print(f"  {cat}: {desc}")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if chr(key) in categories:
                if chr(key) == "x":
                    cv2.destroyAllWindows()
                    exit(0)
                elif chr(key) == "s":
                    break
                else:
                    move_image(image_path, categories[chr(key)], base_path)
                    print(f"Moved {image} to {categories[chr(key)]}")
                    break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    label_images(sorted_image_base_path)
