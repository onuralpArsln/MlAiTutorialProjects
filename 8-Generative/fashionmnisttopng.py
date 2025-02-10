import os
import numpy as np
from PIL import Image
import os
import shutil
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load Fashion MNIST dataset
(train_images, train_labels), _ = tf.keras.datasets.fashion_mnist.load_data()
folder_name = "fashion_mnist_images"
os.makedirs(folder_name, exist_ok=True)
counter=0

# Save all images as PNG
for i, img in enumerate(train_images):
    image = Image.fromarray(img)  # Convert NumPy array to PIL image
    image.save(os.path.join(folder_name, f"image_{i}.png"))
    counter+=1
print(counter)
# Zip the folder
shutil.make_archive(folder_name, 'zip', folder_name)