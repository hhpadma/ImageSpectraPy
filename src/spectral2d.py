import os
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


@dataclass
class ImageAnalyzer:
    img: np.ndarray
    image_length: float = 1  # Default to 1 if not provided
    threshold: float = 0.05  # Default to 0.05 if not provided

    def subtract_mean(self):
        return self.img - np.mean(self.img)

    def compute_periodogram(self):
        self.img = self.subtract_mean()
        m, n = self.img.shape
        F_X = np.fft.fft2(self.img)/(m*n)
        I_pq = m*n*np.abs(F_X)**2
        I_pq_centered = np.fft.fftshift(I_pq)
        I_pq_centered[I_pq_centered < self.threshold] = 0
        return I_pq_centered

    def compute_radial_spectra(self, I_pq):
        m, n = I_pq.shape
        scale_factor = self.image_length/m
        I_pq = I_pq*np.var(I_pq)
        I_pq = I_pq / np.mean(I_pq)
        P, Q = np.meshgrid(np.arange(-m//2, m//2),
                           np.arange(-n//2, n//2), indexing='ij')
        mask = ((P == 0) & (Q < 0)) | \
               ((1 <= P) & (P < m//2) & (-n//2 <= Q) & (Q < n//2)) | \
               ((P == m//2) & (0 <= Q) & (Q <= n//2))
        R = np.sqrt(P**2 + Q**2) * mask
        r_intervals = np.arange(0, int(np.ceil(np.max(R))) + 1, 1)
        R_spectra, _ = np.histogram(
            R.ravel(), bins=r_intervals, weights=I_pq.ravel())
        R_spectra_counts, _ = np.histogram(R.ravel(), bins=r_intervals)
        non_zero_counts = np.where(R_spectra_counts > 0, R_spectra_counts, 0.1)
        R_spectra /= non_zero_counts
        r_intervals = r_intervals * scale_factor if np.any(r_intervals) else 0
        return r_intervals, R_spectra

    def compute_angular_spectra(self, I_pq):
        m, n = I_pq.shape
        I_pq = I_pq/np.var(I_pq)
        I_pq_scaled = I_pq / np.mean(I_pq)
        P, Q = np.meshgrid(np.arange(-m//2, m//2),
                           np.arange(-n//2, n//2), indexing='ij')
        mask = ((P == 0) & (Q < 0)) | \
               ((1 <= P) & (P < m//2) & (-n//2 <= Q) & (Q < n//2)) | \
               ((P == m//2) & (0 <= Q) & (Q <= n//2))
        Theta = np.deg2rad(np.arctan2(P, Q) * 180 / np.pi)
        theta_intervals_deg = np.arange(-5, 180, 10)
        theta_intervals = np.deg2rad(theta_intervals_deg)
        Theta_spectra, _ = np.histogram(
            Theta.ravel(), bins=theta_intervals, weights=I_pq_scaled.ravel())
        Theta_spectra_counts, _ = np.histogram(
            Theta.ravel(), bins=theta_intervals)
        non_zero_counts = np.where(
            Theta_spectra_counts > 0, Theta_spectra_counts, 1)
        Theta_spectra = Theta_spectra/non_zero_counts
        return theta_intervals_deg, Theta_spectra


class ImageAnalyzerVisualizer:
    def __init__(self, analyzer: ImageAnalyzer, image_name: str):
        self.analyzer = analyzer
        self.image_length = analyzer.image_length
        self.image_name = image_name

    def plot_Ipq_center(self, I_pq, ax):
        m, n = I_pq.shape
        center_I_pq = I_pq[m//2-16:m//2+16, n//2-16:n//2+16]
        k = center_I_pq.shape[0]
        cax = ax.imshow(center_I_pq, extent=(-k//2, k//2-1, -k//2, k//2-1),
                        origin='lower', aspect='equal', cmap='gray_r')
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=9))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=9))
        ax.set_title("Periodogram", fontsize=10, pad=10,)

    def plot_R_spectra(self, r_intervals, R_spectra, ax):
        ax.plot(r_intervals[:-1], R_spectra, color='black', linewidth=0.5)
        ax.set_xlim([0, self.analyzer.image_length*10/512])
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_xticks(np.arange(0, self.analyzer.image_length*10/512 + 0.01, 0.01))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        ax.set_xlabel("Wavelength (mm)", fontsize=10)
        ax.set_ylabel("Radial spectrum", fontsize=10)

    def plot_Theta_spectra(self, theta_intervals_deg, Theta_spectra, ax):
        ax.plot(theta_intervals_deg[:-1],
                Theta_spectra, color='black', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        ax.set_xlabel(f"Angle ({chr(176)})", fontsize=10)
        ax.set_ylabel("Angular spectrum", fontsize=10)

    def visualize(self):
        I_pq = self.analyzer.compute_periodogram()
        r_intervals, R_spectra = self.analyzer.compute_radial_spectra(I_pq)
        theta_intervals_deg, Theta_spectra = self.analyzer.compute_angular_spectra(
            I_pq)

        fig, axs = plt.subplots(2, 2, figsize=(5, 5), dpi=300, gridspec_kw={
                                'width_ratios': [1, 1], 'height_ratios': [1, 1]})
        plt.subplots_adjust(wspace=0.5, hspace=0.25)

        axs[0, 0].imshow(self.analyzer.img, extent=(0, 1, 0, 1), cmap='gray')
        axs[0, 0].set_title("Microscopic Image (mm)", fontsize=10,
                            pad=10,)
        axs[0, 0].xaxis.set_major_locator(plt.MaxNLocator(nbins=2))
        axs[0, 0].yaxis.set_major_locator(plt.MaxNLocator(nbins=2))

        self.plot_Ipq_center(I_pq, axs[0, 1])
        self.plot_R_spectra(r_intervals, R_spectra, axs[1, 0])
        self.plot_Theta_spectra(theta_intervals_deg, Theta_spectra, axs[1, 1])
        # plt.savefig(os.path.join(os.getcwd(), f"{self.image_name}.png"))
        #plt.show()
class ImageAnalyzerTester:
    def __init__(self, analyzer: ImageAnalyzer, expected_peak, expected_peak_r, expected_peak_theta):
        self.analyzer = analyzer
        self.expected_peak = expected_peak
        self.expected_peak_r = expected_peak_r
        self.expected_peak_theta = expected_peak_theta

    def test_frequency(self, I_pq):
        m, n = I_pq.shape
        peak_pq = np.unravel_index(np.argmax(I_pq), I_pq.shape)
        peak_pq_shifted = (peak_pq[0] - m//2, peak_pq[1] - n//2)
        print(f"Found peak at: {peak_pq_shifted}, Expected: {self.expected_peak}")
        assert peak_pq_shifted == self.expected_peak, "Test 1: Frequency Verification Failed"
        print("Test 1: Frequency Verification Passed")

    def test_wavelength(self, r_intervals, R_spectra):
        peak_r_index = np.argmax(R_spectra)
        peak_r = r_intervals[peak_r_index]
        print(f"Found peak at: {peak_r}, Expected: {self.expected_peak_r}")
        assert np.isclose(peak_r, self.expected_peak_r, atol=1e-1), "Test 2: Wavelength Verification Failed"
        print("Test 2: Wavelength Verification Passed")

    def test_angle(self, theta_intervals_deg, Theta_spectra):
        peak_theta_index = np.argmax(Theta_spectra)
        peak_theta = theta_intervals_deg[peak_theta_index]
        print(f"Found peak at: {peak_theta}, Expected: {self.expected_peak_theta}")
        assert peak_theta == self.expected_peak_theta, "Test 3: Angle Verification Failed"
        print("Test 3: Angle Verification Passed")

    def run_tests(self):
        I_pq = self.analyzer.compute_periodogram()
        r_intervals, R_spectra = self.analyzer.compute_radial_spectra(I_pq)
        theta_intervals_deg, Theta_spectra = self.analyzer.compute_angular_spectra(I_pq)

        self.test_frequency(I_pq)
        self.test_wavelength(r_intervals, R_spectra)
        self.test_angle(theta_intervals_deg, Theta_spectra)

 

if __name__ == "__main__":
    # Parameters for signal generation
    m = 512
    n = 512
    p = 5
    q = 5

    # Generate a clean signal

    def generate_signal(p, q, m, n, S, T):
        X_st = np.cos(2 * np.pi * (p * S / m + q * T / n))
        return X_st

    def compute_Xst_with_noise(p, q, m, n, s, t):
        """
        Computes a noisy 2D cosine signal with given parameters.

        Args:
        - p (int): frequency of the cosine signal in the x-direction
        - q (int): frequency of the cosine signal in the y-direction
        - m (int): number of samples in the x-direction
        - n (int): number of samples in the y-direction

        Returns:
        - X_st_noisy (ndarray): a 2D numpy array of shape (m, n) containing the noisy cosine signal
        """
        argument = 2 * np.pi * ((p * s / m) - (q * t / n))
        X_st = np.cos(argument)
        noise = np.random.normal(loc=0, scale=1, size=X_st.shape)
        X_st_noisy = X_st + noise
        return X_st_noisy

    def generate_2d_cosine_wave(m, n, frequency, angle):
        """
        Generates a 2D cosine wave with a given frequency and angle.

        Args:
        - m (int): number of samples in the x-direction
        - n (int): number of samples in the y-direction
        - frequency (float): frequency of the cosine wave
        - angle (float): angle of the cosine wave in degrees

        Returns:
        - wave (ndarray): a 2D numpy array of shape (m, n) containing the cosine wave
        """
        # Convert angle to radians
        angle_rad = np.deg2rad(angle)

        # Generate grid of points
        x = np.linspace(0, 1, m)
        y = np.linspace(0, 1, n)
        X, Y = np.meshgrid(x, y)

        # Generate wave
        wave = np.cos(2 * np.pi * frequency * (X * np.cos(angle_rad) + Y * np.sin(angle_rad)))

        return wave

    s = np.arange(1, m+1)
    t = np.arange(1, n+1)
    S, T = np.meshgrid(s, t)

    # Instantiate analyzer and visualizer with the clean signal
    # X_st = generate_signal(p, q, m, n, S, T)
    #X_st = generate_signal(p, q, m, n, S, T)
    X_st = generate_2d_cosine_wave(m, n, p, 90)
    
    analyzer_clean = ImageAnalyzer(X_st)
    visualizer_clean = ImageAnalyzerVisualizer(
        analyzer_clean, "Simple cosine wave")

    # Visualize and perform tests using the clean signal
    visualizer_clean.visualize()
    tester = ImageAnalyzerTester(analyzer_clean, expected_peak=(-p, -q), expected_peak_r=m*np.sqrt(2)/(2*p), expected_peak_theta=45)
    tester.run_tests()
   