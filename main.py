import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.spectral2d import ImageAnalyzer, ImageAnalyzerVisualizer
from src.preprocessor import prepare_image, remove_noise
# Path to the input images

input_folder = "./input"
output_folder = "./output"

target_size = 512
# Less formal "censoring approach" from Renshaw et al. 1983
threshold = 4/(target_size**2)
image_length = 1.0  # mm

# Check if output folder exists, if not create it
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Iterate through each file in the input folder
for file_name in os.listdir(input_folder):
    # Check if the file is an image
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
        # Load image
        img_path = os.path.join(input_folder, file_name)
        # Loading in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_denoised = remove_noise(img)
        img = prepare_image(img, target_size)

        # Instantiate analyzer and visualizer
        analyzer = ImageAnalyzer(img, image_length, threshold)
        visualizer = ImageAnalyzerVisualizer(analyzer, file_name.split('.')[0])

        # Visualize and save the results
        visualizer.visualize()
        plt.savefig(os.path.join(output_folder,
                    f"{file_name.split('.')[0]}_result.png"))
        plt.close()
