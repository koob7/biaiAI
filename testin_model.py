import os
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

labels = ['dog', 'horse', 'elephant', 'butterfly', 'hen', 'cat', 'cow', 'sheep', 'spider', 'squirrel']
model = models.load_model('animal_classifier.keras')



"""
# Prepare the image
img = cv2.imread('animal_test_image/sheep.jpg')
img = tf.image.resize(img, (256, 256))

#Predict with model
prediction = model.predict(np.array([img])/255)
index = np.argmax(prediction)
print(f'prediction is {labels[index]}')

# Plot the image
plt.imshow(cv2.cvtColor(img.numpy(), cv2.COLOR_BGR2RGB).astype(int))
plt.show()
"""


img_dir = 'animal_test_image'
image_files = os.listdir(img_dir)
for file in image_files:
    img_path = os.path.join(img_dir, file)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))

    # Predict with model
    prediction = model.predict(np.array([img])/255)
    index = np.argmax(prediction)
    print(f'Prediction for {file}: {labels[index]}')

    # Plot the image
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(int))
    plt.title(f'Prediction: {labels[index]}')
    plt.axis('off')

plt.show()


