import os
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to convert Pillow image to OpenCV format
def pil_to_cv2(pil_image):
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    return open_cv_image[:, :, ::-1].copy()

# Function to augment and save images
def augment_image(image, save_path, image_name, datagen, num_augments=5):
    image = np.expand_dims(image, axis=0)
    augmented_images = datagen.flow(image, batch_size=1)

    for i in range(num_augments):
        augmented_image = next(augmented_images)[0].astype('uint8')
        save_filename = os.path.join(save_path, f"{os.path.splitext(image_name)[0]}aug{i}.png")
        cv2.imwrite(save_filename, augmented_image)

# Function to traverse folders and apply augmentation
def augment_images_in_folder(input_folder, output_folder, num_augments=5):
    datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=30
    )

    for root, dirs, files in os.walk(input_folder):
        relative_path = os.path.relpath(root, input_folder)
        save_path = os.path.join(output_folder, relative_path)
        os.makedirs(save_path, exist_ok=True)

        for file_name in files:
            image_path = os.path.join(root, file_name)

            try:
                # Use Pillow to load the image with Unicode support
                pil_image = Image.open(image_path)
                image = pil_to_cv2(pil_image)  # Convert Pillow image to OpenCV format

                # Augment and save images
                augment_image(image, save_path, file_name, datagen, num_augments)
            except Exception as e:
                print(f"Error loading image: {image_path}, Error: {e}")

# Specify input and output folders
input_folder = r'C:/Users/Niranjan/Documents/STUDY/MajorProject/DATAS/OwnData'
output_folder = r'C:/Users/Niranjan/Documents/STUDY/MajorProject/DATAS/AugmentedData'

# Run the augmentation process
augment_images_in_folder(input_folder, output_folder, num_augments=5)