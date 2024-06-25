# Import Data Science Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import itertools
import random
from sklearn.metrics import classification_report, confusion_matrix
# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, compare_historys, walk_through_dir, pred_and_plot
# Tensorflow Libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.python.keras import backend as K
# System libraries
from pathlib import Path
import os

# Define helper function to seed everything
def seed_everything(seed=42):
    # Seed value for TensorFlow
    tf.random.set_seed(seed)
    
    # Seed value for NumPy
    np.random.seed(seed)
    
    # Seed value for Python's random library
    random.seed(seed)
    
    # Force TensorFlow to use single thread
    # Multiple threads are a potential source of non-reproducible results.
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )

    # Make sure that TensorFlow uses a deterministic operation wherever possible
    tf.compat.v1.set_random_seed(seed)

    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), )
    K.set_session(sess)

seed_everything()

# Seed everything for reproducibility
seed_everything()


# Define class for custom data generator inheriting from Sequence
class PyDataset(tf.keras.utils.Sequence):
    def __init__(self, data, labels, batch_size=32, shuffle=True):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        data_batch = [self.data[idx] for idx in indexes]
        label_batch = [self.labels[idx] for idx in indexes]
        return np.array(data_batch), np.array(label_batch)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Function to convert dataset path to DataFrame
def convert_path_to_df(dataset_path):
    image_dir = Path(dataset_path)

    # Get filepaths and labels
    filepaths = list(image_dir.glob("**/*.jpg")) + list(image_dir.glob("**/*.jpeg")) + list(image_dir.glob("**/*.png"))
    labels = [os.path.split(os.path.split(filepath)[0])[1] for filepath in filepaths]

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels into DataFrame
    image_df = pd.concat([filepaths, labels], axis=1)
    return image_df

# Function to check for corrupted images
def check_corrupted_images(dataset_path):
    corrupted_images = []
    for img_path in Path(dataset_path).rglob("*.jpg"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                corrupted_images.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            corrupted_images.append(img_path)
    return corrupted_images

# Function to create and display a bar plot of label distribution
def plot_label_distribution(image_df):
    label_counts = image_df['Label'].value_counts()
    plt.figure(figsize=(15, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, alpha=0.8, palette='pastel')
    plt.title('Distribution of Labels in Image Dataset', fontsize=16)
    plt.xlabel('Label', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.show()

# Function to display random images from the dataset
def display_random_images(image_df, num_images=16):
    random_indices = np.random.randint(0, len(image_df), num_images)
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        img_path = image_df.Filepath.iloc[random_indices[i]]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(image_df.Label.iloc[random_indices[i]])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Function to perform ELA (Error Level Analysis) on images
def compute_ela_cv(path, quality):
    orig_img = cv2.imread(path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    temp_filename = 'temp_file_name.jpeg'
    cv2.imwrite(temp_filename, orig_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    compressed_img = cv2.imread(temp_filename)
    diff = cv2.absdiff(orig_img, compressed_img)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
    return diff

# Function to convert image to ELA image
def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpeg'
    ela_filename = 'temp_ela.png'
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(temp_filename, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    temp_image = cv2.imread(temp_filename)
    ela_image = cv2.absdiff(image, temp_image)
    ela_image = cv2.cvtColor(ela_image, cv2.COLOR_BGR2RGB)
    
    return ela_image

# Function to create and return a random sample image path from a directory
def random_sample(directory):
    items = list(Path(directory).rglob('*.jpg'))
    if items:
        return random.choice(items).as_posix()
    return None

# Define constants
BATCH_SIZE = 64
TARGET_SIZE = (112, 112)

# Define paths
dataset_path = "C:/Users/sjaku/Desktop/ML/archive/raw-img"
walk_through_dir(dataset_path)
# Convert dataset path to DataFrame
image_df = convert_path_to_df(dataset_path)

# Check for corrupted images
corrupted_images = check_corrupted_images(dataset_path)
if corrupted_images:
    print(f"Corrupted images found: {corrupted_images}")

# Display label distribution
plot_label_distribution(image_df)

# Display random images from the dataset
display_random_images(image_df)

# Split dataset into training and testing sets
train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True, random_state=42)

# Data Generators
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

# Create generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=TARGET_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    subset='training'
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=TARGET_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=TARGET_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Load the pretrained model
pretrained_model = EfficientNetB7(
    input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
    include_top=False,
    weights='imagenet',
    pooling='max'
)

pretrained_model.trainable = False

# Build the model
inputs = pretrained_model.input
x = tf.keras.layers.Rescaling(1./255)(inputs)
x = tf.keras.layers.RandomFlip("horizontal")(x)
x = tf.keras.layers.RandomRotation(0.1)(x)
x = tf.keras.layers.RandomZoom(0.1)(x)

x = Dense(256, activation='relu')(pretrained_model.output)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

outputs = Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
checkpoint_path = "animals_classification_model_checkpoint.weights.h5"
checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    monitor="val_accuracy",
    save_best_only=True
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

# Train the model
history = model.fit(
    train_generator,
    epochs=1,
    validation_data=val_generator,
    callbacks=[checkpoint_callback, early_stopping, reduce_lr]
)

# Save the model in HDF5 format
# model.save("animals_classification_model.h5")

# Save the model in TensorFlow SavedModel format
model.save("animals_classification_model.keras")

# Evaluate the model on test data
results = model.evaluate(test_generator, verbose=0)
print(f'Test Loss: {results[0]}')
print(f'Test Accuracy: {results[1]}')




# Predictions
preds = model.predict(test_generator)
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes

# Classification report and confusion matrix
class_labels = list(test_generator.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=class_labels))

# Plot training history
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Display random predictions
random_indices = np.random.choice(len(test_df), 25, replace=False)
plt.figure(figsize=(15, 10))
for i, idx in enumerate(random_indices):
    plt.subplot(5, 5, i + 1)
    img = cv2.imread(test_df['Filepath'].iloc[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(f"True: {test_df['Label'].iloc[idx]}\nPredicted: {class_labels[y_pred[idx]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

print('End of the application')
