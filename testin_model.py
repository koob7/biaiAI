import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, compare_historys, walk_through_dir, pred_and_plot
import os
from pathlib import Path
# Define the labels
labels = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']


# Load the trained model
model = models.load_model('models/animals_classification_model.keras')
from tensorflow.keras.applications.efficientnet import preprocess_input

def select_image():
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return
    
    # Load and preprocess the image
    img = cv2.imread(file_path)
    if img is not None:
        img_resized = cv2.resize(img, (224, 224))  # Use the correct size expected by the model
        img_array = np.array([img_resized])
        img_array = preprocess_input(img_array)  # Apply the same preprocessing as during training
        
        # Make a prediction
        prediction = model.predict(img_array)
        index = np.argmax(prediction)
        prediction_label = labels[index]
        
        # Update the GUI with the selected image and prediction
        display_image(img, prediction_label)


def display_image(img, prediction_label):
    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert the image to a format suitable for displaying in Tkinter
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
     # Update the prediction label
    prediction_label_var.set(f'Prediction: {prediction_label}')   
    # Update the image label
    image_label.config(image=img_tk)
    image_label.image = img_tk
    


# Create the main application window
root = tk.Tk()
root.title("Animal Classification")

# Create and place a button to select an image
select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack()



# Create and place a label to display the prediction
prediction_label_var = tk.StringVar()
prediction_label = tk.Label(root, textvariable=prediction_label_var)
prediction_label.pack()
# Create and place a label to display the selected image
image_label = tk.Label(root)
image_label.pack()
# Run the application
root.mainloop()
