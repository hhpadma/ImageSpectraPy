
# ImageSpectraPy: Image Preprocessing and Spectral Analysis in Python

ImageSpectraPy is a Python toolkit that performs the following functions:

Image Preprocessing:

Removes noise from images.
Converts images to grayscale.
Resizes images to standard dimensions.
Spectral Analysis:

Uses Fast Fourier Transforms (FFT) to transition images into the frequency domain.
Calculates periodograms to visualize the frequency components of images.
Computes radial and angular spectra of images.
The toolkit is designed for users who need to preprocess images and perform spectral analysis.

## Installation

```bash
git clone https://github.com/hhpadma/ImageSpectraPy.git
cd your-repo
pip install -r requirements.txt
```


## File Descriptions

- **main.py**: This script serves as the entry point for executing the program. It handles the workflow by processing input images and saving the analyzed results.
- **preprocessor.py**: Contains functions for image preprocessing, including noise removal, grayscale conversion, resizing, and aspect ratio adjustments.
- **spectral2d.py**: Handles the spectral analysis of images, including Fourier transformations, periodogram calculation, and computation of radial and angular spectra.

## Usage

Clone the Repository:

Clone this repository to your local machine using git clone https://github.com/hhpadma/ImageSpectraPy.git.
Navigate to the cloned repository by running cd ImageSpectraPy.
Set Up the Python Environment:

If you haven't already, install Python on your system. This project requires Python 3.x.
It's recommended to create a virtual environment for this project to avoid conflicts with other projects. Use python -m venv env to create a virtual environment and source env/bin/activate (or env\Scripts\activate on Windows) to activate it.
Install the necessary Python packages using pip install -r requirements.txt.
Prepare Your Images:

Place the images you want to analyze in the input directory. If the directory doesn't exist, create it at the root of the project folder.
Run the Script:

From the root directory of the project, run the script using the command python main.py. This script will process the images located in the input directory and place the results in the output directory.
If you need to specify different directories or any other command-line arguments, adjust the command accordingly. For example: python main.py --input your_input_folder --output your_output_folder.
Review the Results:

Once the script has completed, check the output directory for the processed images and their spectral analysis results.

## Contributing

Instructions on how to contribute to your project. This might include:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

