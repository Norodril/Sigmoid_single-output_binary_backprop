import matplotlib
matplotlib.use('TkAgg') #forcing pop up window for cat image to be recognized

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

model = tf.keras.applications.MobileNetV2(weights="imagenet")

#loads and prepares the image
image_path = keras.utils.get_file(
    "cat.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"
)

img = Image.open(image_path).reside((224, 224))
img_array = np.array(img)

#preprocess for the model
x = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top = 3)[0]

#show results
plt.imshow(img)
plt.title("f"Prediction: {decoded[0][1]} ({decoded[0][2]*100:.2f}%)")"
plt.axis("off")
plt.show()

print("Top 3 predictions:")
for(_, label, prob) in decoded:
print(f"{label}: {prob*100:.2f}%")