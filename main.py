import cv2
import numpy as np
import csv
import os

# Function to process image and extract edge points
def process_image_and_save_points(image_path, csv_writer, t_lower=130, t_upper=225):
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize image to original size
    img_resized = cv2.resize(img, (256, 256))

    # Apply Canny Edge detection
    edge = cv2.Canny(img_resized, t_lower, t_upper)

    # Find edge points
    points = np.argwhere(edge > 0)  # Get indices of non-zero elements

    # Scale the points back to the original image size
    points_scaled = points * 2  # Resize factor is 2 (original size: 256x256, display size: 512x512)

    # Save edge points to CSV
    points_array = points_scaled.flatten().tolist()
    csv_writer.writerow([image_path] + points_array)

# Directory containing images
image_dir = "D:/ML Projects/Dataset/testing/parkinson/"

# Path to the CSV file to save edge points
csv_file_path = "D:/ML Projects/edge_points.csv"

# Create/Open the CSV file for writing
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Image_Path', 'Edge_Points'])  # Write header

    # Process each image in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):  # Process only PNG files, adjust as needed
            image_path = os.path.join(image_dir, filename)
            process_image_and_save_points(image_path, csv_writer)

print("Edge points saved to:", csv_file_path)
