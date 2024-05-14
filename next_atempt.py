import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras import models
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


cpus = tf.config.experimental.list_physical_devices('CPU')
print(cpus)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = 'raw-img'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

"""
for image_class in os.listdir (data_dir):
    print(image_class)
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('bledny format obrazu {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('issue with image {}'.format((image_path)))
"""

data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    batch_size =12,
    )
data = data.map(lambda x, y: (x / 255, y))

labels = ['dog', 'horse', 'elephant', 'butterfly', 'hen', 'cat', 'cow', 'sheep', 'spider', 'squirrel']
data_iterator = data.as_numpy_iterator()
"""
for k in range(3):
    batch = data_iterator.next()
    fig, ax = plt.subplots(ncols=6, nrows=5, figsize=(10, 10))
    for i in range(5):
        for j in range(6):
            idx = i * 6 + j
            if idx < len(batch[0]):
                ax[i, j].imshow(batch[0][idx])
                ax[i, j].set_title(labels[batch[1][idx]])
                ax[i, j].axis('off')
    plt.show()
"""
print(len(data)) #ilość paczek
train_size = int(len(data)*0.7)#nauka
val_size = int(len(data)*0.2)+1#ocena w szkoleniu
test_size = int(len(data)*0.1)+1#ocena po szkoleniuj
print(train_size+val_size+test_size)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

"""
model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape = (256,256,3)))#convolutional layer - filtruje cechy obrazu - 32 neurony, 3x3convolutional matrix, funkcja aktywacyjna, zdjęcie 32x32xRGB
model.add(MaxPooling2D())#pooling layer - ogranicza liczbę informacji na obrazie

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()


logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs = 1, validation_data=val, callbacks=[tensorboard_callback])

"""
model = models.Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape = (256,256,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
model.fit(train, epochs = 10, validation_data=val, callbacks=[tensorboard_callback])
#print(f"loss:{loss}")
#print(f"Accurady: {accuracy}")
model.save('animal_classifier.keras')
"""
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()


pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(pre.result(), re.result(), acc.result())

img = cv2.imread('test_image/plane.jpg')
resize = tf.image.resize(img, (256, 256))
yhat = model.predict(np.expand_dims(resize/255, 0))
print(labels[yhat.astype(int)])
plt.imshow(cv2.cvtColor(resize.numpy().astype(int)))
plt.show()
"""