import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from io import BytesIO
import zipfile

# Function to process the image and extract rice grains
def process_image(image):
    image_info = []  # List to store image information (size in pixels and file size)
    
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
    min_area = 3000  # Adjust this based on your image
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

    # Display each cropped rice grain and collect selection information
    for i, contour in enumerate(filtered_contours):
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the region
        cropped_image = image[y:y + h, x:x + w]

        # Encode image to bytes
        _, buffer = cv2.imencode('.png', cropped_image)
        image_size_kb = len(buffer) / 1024  # Size in KB
        image_size_mb = image_size_kb / 1024  # Size in MB

        image_info.append((i, w, h, image_size_kb, image_size_mb, cropped_image))

    return image_info

# Function to save selected images to a zip file with higher quality
def save_to_zip(selections):
    with tempfile.TemporaryDirectory() as tmpdirname:
        zip_path = os.path.join(tmpdirname, "selected_rice_grains.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for i, (name, image) in enumerate(selections):
                image_filename = f"{name}.png"
                image_path = os.path.join(tmpdirname, image_filename)
                # Save the image with higher quality
                cv2.imwrite(image_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])  # Compression quality set to maximum (0-9)
                zipf.write(image_path, image_filename)
        with open(zip_path, 'rb') as f:
            zip_data = f.read()
    return zip_data

# Streamlit app
st.set_page_config(layout="wide")  # Utilize full screen
st.title("Rice Grain Extraction")

# Add background color
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Instructions
with st.expander("Instructions", expanded=True):
    st.markdown("1. Upload an image containing rice grains.")
    st.markdown("2. Select the rice grains you want to extract.")
    st.markdown("3. Click the 'Extract' button to download the selected rice grains as a ZIP file.")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# If an image is uploaded
if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the original image
    st.image(image, caption='Original Image', use_column_width=True)

    # Process the image and get selected images
    image_info = process_image(image)

    # Define size filter options
    size_filter = st.selectbox("Select size range to display images:",
                               ["All", "0-0.03 MB", "0.031-0.06 MB", "0.061-0.09 MB", ">0.09 MB"])

    # Filter images based on selected size range
    if size_filter == "0-0.03 MB":
        filtered_images = [info for info in image_info if info[4] <= 0.03]
    elif size_filter == "0.031-0.06 MB":
        filtered_images = [info for info in image_info if 0.031 <= info[4] <= 0.06]
    elif size_filter == "0.061-0.09 MB":
        filtered_images = [info for info in image_info if 0.061 <= info[4] <= 0.09]
    elif size_filter == ">0.09 MB":
        filtered_images = [info for info in image_info if info[4] > 0.09]
    else:
        filtered_images = image_info

    # Display filtered images
    max_images_per_row = 4
    checkboxes = []
    select_all = st.checkbox("Select All")

    for i, (idx, w, h, size_kb, size_mb, cropped_image) in enumerate(filtered_images):
        if i % max_images_per_row == 0:
            col = st.columns(max_images_per_row)  # Create a new row
        with col[i % max_images_per_row]:
            st.image(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB), caption=f"Rice Grain {idx + 1}")
            st.write(f"Size: {w} x {h} pixels")
            st.write(f"File Size: {size_kb:.2f} KB ({size_mb:.2f} MB)")
            checkbox = st.checkbox(f"Select Rice Grain {idx + 1}", key=f"select_{idx}", value=select_all)
            checkboxes.append((checkbox, (f"Rice Grain {idx + 1}", cropped_image)))

    # Update selections based on checkboxes
    selections = [img_info for selected, img_info in checkboxes if selected]

    st.write(f"Extracted and displayed {len(filtered_images)} rice grains.")

    # Button to extract selected images
    if st.button("Extract"):
        if selections:
            zip_data = save_to_zip(selections)
            st.download_button(label="Download ZIP", data=zip_data, file_name="selected_rice_grains.zip", mime="application/zip")
        else:
            st.warning("No rice grains selected. Please select at least one rice grain.")
