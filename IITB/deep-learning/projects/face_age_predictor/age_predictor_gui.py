# ===================================================================
# Python GUI Script: Predict Age from an Image
# ===================================================================
# This script creates a simple GUI to load the trained model,
# upload an image, and display the predicted age.

import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Canvas
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os

# --- 1. CONFIGURATION ---
MODEL_PATH = "age_prediction_model.h5"  # Path to your saved model
IMAGE_SIZE = (128, 128)                 # Must be the same size as used during training
BG_COLOR = "#2c3e50"
TEXT_COLOR = "#ecf0f1"
BUTTON_COLOR = "#3498db"
FONT_STYLE = ("Helvetica", 12)
TITLE_FONT_STYLE = ("Helvetica", 16, "bold")

# Global variable for the model
model = None

# --- 2. CORE FUNCTIONS ---

def load_model():
    """Loads the pre-trained Keras model and updates the status label."""
    global model
    status_label.config(text="Loading model, please wait...")
    
    if not os.path.exists(MODEL_PATH):
        status_label.config(text=f"Error: Model file not found at '{MODEL_PATH}'")
        return

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        model.summary()  # Prints summary to console
        status_label.config(text="Model loaded successfully. Please upload an image.")
    except Exception as e:
        status_label.config(text=f"Error loading model: {e}")

def predict_age(image_path):
    """
    Loads an image, preprocesses it, and predicts the age using the loaded model.
    
    Args:
        image_path (str): Path to the input image.

    Returns:
        tuple: A tuple containing the predicted age (int) and the original PIL Image object.
    """
    if model is None:
        result_label.config(text="Model is not loaded.")
        return None, None

    try:
        # --- Image Preprocessing ---
        img = Image.open(image_path).convert('RGB')
        
        # Create a display-friendly version of the image
        display_img = img.copy()
        display_img.thumbnail((300, 300)) # Resize for display
        
        # Preprocess for the model
        img_resized = img.resize(IMAGE_SIZE)
        img_array = np.array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # --- Make Prediction ---
        predicted_age = model.predict(img_batch)[0][0]
        
        return int(round(predicted_age)), display_img

    except Exception as e:
        status_label.config(text=f"Error during prediction: {e}")
        return None, None

def upload_and_predict():
    """Handles the image upload, prediction, and GUI update."""
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    
    if not file_path:
        return # User cancelled the dialog

    # Predict age
    predicted_age, display_img = predict_age(file_path)

    if predicted_age is not None and display_img is not None:
        # Update the image display
        img_tk = ImageTk.PhotoImage(display_img)
        image_label.config(image=img_tk)
        image_label.image = img_tk  # Keep a reference to avoid garbage collection

        # Update the result label
        result_label.config(text=f"Predicted Age: ~{predicted_age} years")
        status_label.config(text=f"Prediction complete for: {os.path.basename(file_path)}")

# --- 3. GUI SETUP ---
# Create the main window
root = tk.Tk()
root.title("Age Predictor")
root.geometry("400x500")
root.configure(bg=BG_COLOR)
root.resizable(False, False)

# Create a main frame
main_frame = Frame(root, bg=BG_COLOR, padx=20, pady=20)
main_frame.pack(expand=True, fill=tk.BOTH)

# Title Label
title_label = Label(main_frame, text="Age Prediction from Image", font=TITLE_FONT_STYLE, bg=BG_COLOR, fg=TEXT_COLOR)
title_label.pack(pady=(0, 20))

# Image Display Label
image_label = Label(main_frame, bg=BG_COLOR)
image_label.pack(pady=10)

# Upload Button
upload_button = Button(main_frame, text="Upload Image", command=upload_and_predict, font=FONT_STYLE, bg=BUTTON_COLOR, fg="white", relief=tk.FLAT, padx=10, pady=5)
upload_button.pack(pady=10)

# Result Label
result_label = Label(main_frame, text="Predicted Age: --", font=TITLE_FONT_STYLE, bg=BG_COLOR, fg=TEXT_COLOR)
result_label.pack(pady=10)

# Status Label
status_label = Label(main_frame, text="Initializing...", font=FONT_STYLE, bg=BG_COLOR, fg="#bdc3c7")
status_label.pack(side=tk.BOTTOM, fill=tk.X)

# --- 4. START THE APPLICATION ---
# Load the model after the GUI window is created
root.after(100, load_model)

# Start the Tkinter event loop
root.mainloop()
