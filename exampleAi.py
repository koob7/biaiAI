import os
from tqdm import tqdm
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
#przygotowanie danych
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

training_images, testing_images = training_images/255, testing_images/255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

#wyświetlenie próbki danych

print(len(training_images))

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()


print("Rozmiar danych treningowych (images):", training_images.shape)
print("Rozmiar danych treningowych (labels):", training_labels.shape)

print(training_labels)

#początek sieci neuronowej
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape = (32,32,3)))#convolutional layer - filtruje cechy obrazu - 32 neurony, 3x3convolutional matrix, funkcja aktywacyjna, zdjęcie 32x32xRGB
model.add(layers.MaxPooling2D((2,2)))#pooling layer - ogranicza liczbę informacji na obrazie
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())#wypłaszczamy to co dostaliśmy (układamy macierz w wiersz)
model.add(layers.Dense(64, activation='relu'))#dense layers
model.add(layers.Dense(10, activation='softmax'))#10 - dla każdej klasy, łaczymy uzyskane sumy do 100%
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10, validation_data = (testing_images, testing_labels)) #epoch- ile razy sieć zobaczy ten sam obraz

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"loss:{loss}")
print(f"Accurady: {accuracy}")
model.save('image_classifier.keras')


"""
model = models.load_model('image_classifier.keras')

img = cv.imread('test_image/deer.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img])/255)
index = np.argmax(prediction)
print(f'prediction is {class_names[index]}')
"""
plt.show()
