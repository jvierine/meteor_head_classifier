from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.config.experimental.list_physical_devices()
import cv2

# Set directory path to dataset
dataset_path =r"C:\Users\ragav\meteor_head_classifier\data1\pansy\sorted_images2"  # Dataset path

# Load images from directories, split into training and validation sets
batch_size = 32
img_size = (256, 256)

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# Get class names from folders
class_names = train_ds.class_names
print(f"Class names: {class_names}")


# Load the trained model
model = load_model("finetuned.keras")

def label_image(img_path):
    # Load an image for prediction
    #img_path = "path/to/image.png"
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    print(f"Predicted class: {predicted_class}")

    # plot 
    img = cv2.imread(img_path)
    # Scale factor (increase by 2x)
    scale = 2.0  
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    resized = cv2.resize(img, (width, height))
    cv2.imshow("Label Image", resized)
    key = cv2.waitKey(0) & 0xFF
    if chr(key) in ["n","x"]:
        if chr(key) == "n":
            cv2.destroyAllWindows()
            return
        if chr(key) == "x":
            cv2.destroyAllWindows()
            exit(0)

        
        
    
#    
 #   cv2.imshow("Label Image", img)



import glob


fl=glob.glob(r"C:\Users\ragav\code\meteor_head_classifier\data1\pansy\cnn_images\cnn*.png")
np.random.shuffle(fl)
for f in fl:
    print(f)
    label_image(f)
    
