import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the image in grayscale
img = cv2.imread('src.png', 0)

# Check if the image was loaded successfully
if img is None:
    print("Error: Could not load image.")
    os.exit(1)

fig = plt.figure(figsize=(15, 5))

# Perform 2D FFT
# np.fft.fft2 computes the 2D discrete Fourier Transform
# np.fft.fftshift shifts the zero-frequency component to the center of the spectrum
f_transform = np.fft.fft2(img)
f_transform_shifted_init = np.fft.fftshift(f_transform)
f_transform_shifted = f_transform_shifted_init 
def update(val):
    global f_transform_shifted
    f_transform_shifted = val * f_transform_shifted_init * val
    fig.canvas.draw_idle()

# Calculate the magnitude spectrum for visualization
magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))

# Perform 2D Inverse FFT
# np.fft.ifftshift shifts the zero-frequency component back to the top-left
# np.fft.ifft2 computes the 2D inverse discrete Fourier Transform
# np.abs() is used to get the real part of the image, as IFFT can return complex numbers
f_ishift = np.fft.ifftshift(f_transform_shifted)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Display the original image, magnitude spectrum, and reconstructed image

axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
freq_slider = plt.Slider(
    ax=axfreq,
    label='Crop (freq)',
    valmin=0.0,
    valmax=1,
    valinit=.5,
)

freq_slider.on_changed(update)

plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(img_back, cmap='gray')
plt.title('Reconstructed Image'), plt.xticks([]), plt.yticks([])

plt.ion()
plt.show()