
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

## Methodology
Two-dimensional spectral analysis is pivotal in elucidating both the intricate structures and diverse scales of patterns inherent in spatial data, offering a nuanced perspective that is indispensable in environmental and ecological research [4]. Integral to this method is the periodogram, a spectral component that efficaciously distributes variance among various frequencies, thereby highlighting periodicities and facilitating the discernment of large-scale patterns and directional trends [3][5].

Initiating the process is the Fourier analysis of spatial data, emphasizing the crucial role of the Fast Fourier Transform (FFT) for computational efficiency, particularly with large datasets [3]. Further sophistication in the analysis is introduced through the radial and angular spectra, methodologies that distill the periodogram's information into more accessible forms. The radial spectrum aggregates data over concentric rings, revealing dominant wave numbers, while the angular spectrum, subdivided into precise angular intervals, underscores the directional aspect of the data [4]. This intricate approach not only augments the intelligibility of spatial data but also permits the identification of subtle, nuanced patterns and anisotropies potentially obscured within the raw data or Cartesian spectrum [3][4].

The methodology adopted for this spectral analysis, as informed by [3] and [4], begins with pivotal pre-processing stages detailed in [1]. These stages include the transformation of original images to grayscale and the careful reduction of noise, enhancing the visibility of features such as cracks. Additionally, images are adjusted to square dimensions by cropping excess areas from the longer side, ensuring that the primary features of interest are retained. Subsequently, images are resized to dimensions — a power of 2 — that suit the user's requirements and meet the FFT algorithm's prerequisites. This specific resizing is critical as it maintains the integrity of the image's original features while facilitating the FFT's computational efficiency and precision.
Once pre-processed, the images undergo an essential mean correction to center the data around zero, eliminating any bias that a non-zero mean might introduce and ensuring a more accurate frequency analysis during the FFT process. This mean-corrected data is then subjected to the FFT algorithm, leading to the derivation of the periodogram and the subsequent extraction of Ipq values. These values represent the contribution of each frequency pair (p, q) to the overall variance in the data, offering a spectral perspective of the spatial structure inherent within.
The computation of the radial spectrum is an intricate process that transforms the Cartesian spectrum, represented as Ipq, into a polar coordinate system. This transformation is essential for accurately identifying the directional components and scales of patterns present in the data. It involves calculating Grθ = Ipq, where r is the radial distance from the origin (computed as to \sqrt{p^2+q^2} ) and θ is the angular component (computed as \tan^{-1}{\frac{p}{q}}.). These Ipq values are then carefully analyzed within a specified range, normalized to a mean of unity, and classified into specific radial distances, facilitating the construction of the R-spectrum.

The integration of wavelength computation, as informed by [5], constitutes a crucial enhancement to this analytical methodology. This adaptation involves the conversion of all identified wavenumbers in the radial spectrum into their corresponding wavelengths. This conversion is executed by dividing the image size by each wavenumber, a technique that proves consistent across images of various sizes, provided they are at least 3–5 times the size of the pattern under investigation. To transpose the computed wavelengths into a real-world scale, a critical calibration step is employed: each wavelength derived from the wavenumber is multiplied by a specific scale factor. This scale factor is contingent upon the original dimensions of the analyzed image, thereby anchoring the computational analysis in the tangible parameters of the sampled environment. This meticulous calibration ensures that the spectral analysis outputs are directly interpretable in terms of real-world spatial dimensions, enhancing the practical applicability and relevance of the results.

This comprehensive methodology, drawing upon various scholarly sources, provides a robust and nuanced approach to the spectral analysis of spatial data, addressing the complexities and subtleties involved in image analysis within environmental studies.

[1]	H. Deng et al., “Crack Patterns of Environmental Plastic Fragments,” Environ. Sci. Technol., vol. 56, no. 10, pp. 6399–6414, May 2022, doi: 10.1021/acs.est.1c08100.
[2]	J. van de Koppel, M. Rietkerk, N. Dankers, and P. M. J. Herman, “Scale‐Dependent Feedback and Regular Spatial Patterns in Young Mussel Beds,” Am. Nat., vol. 165, no. 3, pp. E66–E77, Mar. 2005, doi: 10.1086/428362.
[3]	E. Renshaw and E. D. Ford, “The Interpretation of Process from Pattern Using Two-Dimensional Spectral Analysis: Methods and Problems of Interpretation,” Appl. Stat., vol. 32, no. 1, p. 51, 1983, doi: 10.2307/2348042.
[4]	E. Renshaw and E. D. Ford, “The description of spatial pattern using two-dimensional spectral analysis,” Vegetatio, vol. 56, no. 2, pp. 75–85, Jun. 1984, doi: 10.1007/BF00033049.
[5]	P. Couteron and O. Lejeune, “Periodic spotted patterns in semi-arid vegetation explained by a propagation-inhibition model: Periodic spotted vegetation,” J. Ecol., vol. 89, no. 4, pp. 616–628, Aug. 2001, doi: 10.1046/j.0022-0477.2001.00588.x.

