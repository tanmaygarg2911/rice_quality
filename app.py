import cv2
import os
import streamlit as st
import numpy as np
import pandas as pd


# Function to process the image and extract rice grains
def process_image(image):
    qualities = []  # List to store image qualities
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply adaptive thresholding (using mean method)
    thresh_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY_INV, blockSize=15, C=2)

    # Find contours
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    min_area = 4500  # Adjust this based on your image
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

    # Display each cropped rice grain and collect quality information
    max_images_per_row = 4  # Define the maximum number of images per row
    for i, contour in enumerate(filtered_contours):
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the region
        cropped_image = image[y:y + h, x:x + w]

        # Display the cropped rice grain
        if i % max_images_per_row == 0:
            col = st.columns(max_images_per_row)  # Create a new row
        col[i % max_images_per_row].image(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB), caption=f"Rice Grain {i + 1}")

        # Quality selection
        quality = st.radio(f"Select quality for Rice Grain {i + 1}", ("Good", "Average", "Bad"))
        qualities.append(quality)

    st.write(f"Extracted and displayed {len(filtered_contours)} rice grains.")
    return qualities


# Streamlit app
st.title("Rice Grain Extraction")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Define excel_file_path
excel_file_path = None

# If an image is uploaded
if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the original image
    st.image(image, caption='Original Image', use_column_width=True)

    # Process the image and get qualities
    qualities = process_image(image)

    # Button to submit
    if st.button("Submit"):
        # Create a DataFrame to store image name and quality
        image_names = [f"Rice Grain {i + 1}" for i in range(len(qualities))]
        df = pd.DataFrame({"Image Name": image_names, "Quality": qualities})

        # Save Excel file locally
        excel_file_path = "rice_grain_quality.xlsx"
        df.to_excel(excel_file_path, index=False)
        st.success(f"Excel file saved locally as '{excel_file_path}'")

# Button to download Excel file
if excel_file_path:
    if st.button("Download Excel"):
        st.download_button(label="Click to download", data=open(excel_file_path, "rb"),
                           file_name="rice_grain_quality.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
