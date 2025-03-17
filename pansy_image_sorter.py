import os
import shutil
import cv2

image_directory="/data1/pansy/cnn_images/cnn*.png"
sorted_image_base_path = "/data1/pansy/sorted_images"  # Folder to store sorted images

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
    import glob
    images=glob.glob(image_directory)
    #print(images)
    #images = [f for f in os.listd
    ir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    for image in images:
        image_path = image#osimage#.path.join(image_folder, image)
        img = cv2.imread(image_path)
        # Scale factor (increase by 2x)
        scale = 4.0  
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        resized = cv2.resize(img, (width, height))
        cv2.imshow("Label Image", resized)
        print(f"Label the image {image} by pressing 1-6 on your keyboard")
        
        while True:
            print(f"Label the image {image} by pressing 1-6 on your keyboard")
            for cat in categories.keys():
                print("%s %s"%(cat,categories[cat]))
            key = cv2.waitKey(0) & 0xFF
            if chr(key) in categories:
                if chr(key) == "x":
                    exit(0)
                if chr(key) == "s":
                    break
                move_image(image_path, categories[chr(key)], base_path)
                print(f"Moved {image} to {categories[chr(key)]}")
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    label_images(sorted_image_base_path)
