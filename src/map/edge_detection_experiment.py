import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def update_image(value=None):
    # Get current threshold values from sliders
    low_threshold = low_slider.get()
    high_threshold = high_slider.get()

    # Update labels to show current threshold values
    low_label.config(text=f"Low Threshold: {low_threshold}")
    high_label.config(text=f"High Threshold: {high_threshold}")

    # Perform Canny edge detection
    edged = cv2.Canny(gray, low_threshold, high_threshold)

    # Display the edge-detected image
    img = Image.fromarray(edged)
    img_tk = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.image = img_tk   # Keep a reference to avoid garbage collection

# Load the image
image_path = '../images/maps_page1_1.jpeg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a tkinter root window
root = tk.Tk()
root.title('Edge Detection Experiment')

# Create a frame to hold the scales
frame = ttk.Frame(root)
frame.pack()

# Create sliders for adjusting thresholds
low_slider_default = 50
low_slider = ttk.Scale(frame, from_=0, to=255, orient='horizontal', length=300)
low_slider.set(low_slider_default)      # Initial value
low_slider.pack()

high_slider_default = 150
high_slider = ttk.Scale(frame, from_=0, to=255, orient='horizontal', length=300)
high_slider.set(high_slider_default)    # Initial value
high_slider.pack()

# Add labels to display slider values
low_label = ttk.Label(frame, text=f"Low Threshold: {low_slider_default}")
low_label.pack()

high_label = ttk.Label(frame, text=f"High Threshold: {high_slider_default}")
high_label.pack()

# Set the comand for the sliders after both sliders are created
low_slider.config(command=update_image)
high_slider.config(command=update_image)

# Add a scrollbar for the image
scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas = tk.Canvas(root, width=image.shape[1], height=image.shape[0], yscrollcommand=scrollbar.set)
canvas.pack(side=tk.LEFT)

scrollbar.config(command=canvas.yview)
canvas.config(scrollregion=(0, 0, image.shape[1], image.shape[0]))

# Display the initial edge-detected image
update_image()

# Run the Tkinder event loop
root.mainloop()

# Clean up
cv2.destroyAllWindows()
