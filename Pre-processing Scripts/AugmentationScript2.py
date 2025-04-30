import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread(r'C:\Users\Niranjan\Documents\STUDY\MajorProject\DATAS\OwnData\ಕ್ಷ1.jpg')

# Resize the image to 28x28 pixels
image = cv2.resize(image, (28, 28))

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Reshape the image to (1, height, width, 1) because ImageDataGenerator expects a batch of images
gray_image = np.expand_dims(gray_image, axis=(0, -1))

# Create an instance of ImageDataGenerator with the desired augmentations
datagen = ImageDataGenerator(
    width_shift_range=0.2,    # Horizontal shift (20% of the image width)
    height_shift_range=0.2,   # Vertical shift (20% of the image height)
    rotation_range=30,        # Rotation (up to 30 degrees)
    brightness_range=[0.8, 1.2]  # Brightness adjustment (dim - 80% to 120% of the original brightness)
)

# Directory to save the augmented images
output_dir = 'augmented_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Apply the augmentations and save the images
augmented_images = datagen.flow(gray_image, batch_size=1)

# Generate and save 10 augmented images
for i in range(10):
    augmented_image = next(augmented_images)[0].astype('uint8').reshape(28, 28)
    output_path = f"{output_dir}/augmented_image_{i + 1}.png"
    cv2.imwrite(output_path, augmented_image)

# Display one of the augmented images
plt.imshow(augmented_image, cmap='gray')
plt.axis('off')
plt.show()