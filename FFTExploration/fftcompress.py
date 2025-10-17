import marimo

__generated_with = "unknown"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    import urllib
    return cv2, mo, np, plt, urllib


@app.cell
def _(cv2, np, urllib):

    req = urllib.request.urlopen('https://media.istockphoto.com/id/981616386/photo/deserted-death-valley-in-the-desert.jpg?s=612x612&w=0&k=20&c=cOE5bonzhVBpt7fscAeVfubP9AFa-0haEdosg8uyDCY=')
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, 0)

    imgc = cv2.imdecode(arr, -1)
    return img, imgc


@app.cell
def _(mo):
    crop_slider = mo.ui.slider(start = 0, stop = 1, label="crop", step=0.01, value=.6)
    return (crop_slider,)


@app.cell
def _(crop_slider, img, np):
    # Perform 2D FFT
    # np.fft.fft2 computes the 2D discrete Fourier Transform
    # np.fft.fftshift shifts the zero-frequency component to the center of the spectrum
    f_transform = np.fft.fft2(img)
    f_transform_shifted_init = np.fft.fftshift(f_transform)
    f_transform_shifted = f_transform_shifted_init 

    mask = np.zeros_like(img)
    print(mask.shape)
    ylim = mask.shape[0]/2 * crop_slider.value
    xlim = mask.shape[1]/2 * crop_slider.value

    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            to_center = (row - mask.shape[0]/2, col - mask.shape[1]/2)
            dist = np.sqrt(to_center[0] * to_center[0] + to_center[1] * to_center[1])
            if abs(to_center[0]) < ylim and abs(to_center[1]) < xlim:
                mask[row][col] = 1
            pass
    # Calculate the magnitude spectrum for visualization
    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted)) * mask


    f_transform_masked = f_transform_shifted*mask

    scale = 0.0000035*256

    f_transform_real = (f_transform_masked.real * scale).astype(np.int16)
    f_transform_imag = (f_transform_masked.imag * scale).astype(np.int16)
    print(f"RMx: {f_transform_real.max()} IMx: {f_transform_imag.max()}")
    print(f"RMn: {f_transform_real.min()} IMn: {f_transform_imag.min()}")
    masked_image_size = (int(xlim)*2)*(int(ylim)*2)

    masked_bytes = masked_image_size*2 *np.dtype(np.int16).itemsize
    grayscale_img_bytes = img.shape[0] * img.shape[1]

    print(f"Masked FFT Data Size {masked_bytes} - {masked_bytes/256} LoRa Packets")
    print("Original Color Size", grayscale_img_bytes*3)
    print("Original Grayscale Size", grayscale_img_bytes)
    print(f"{100*masked_bytes/(grayscale_img_bytes*3):.2} % of original")

    f_transform_back = (f_transform_real.astype(np.float32)/scale) + (f_transform_imag.astype(np.float32)/scale)*1j

    # Perform 2D Inverse FFT
    # np.fft.ifftshift shifts the zero-frequency component back to the top-left
    # np.fft.ifft2 computes the 2D inverse discrete Fourier Transform
    # np.abs() is used to get the real part of the image, as IFFT can return complex numbers
    f_ishift = np.fft.ifftshift(f_transform_back)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return f_transform_masked, img_back, magnitude_spectrum, mask, xlim, ylim


@app.cell
def _(f_transform_masked):
    f_transform_masked[0:100][0:100]
    return


@app.cell
def _(
    crop_slider,
    img,
    img_back,
    imgc,
    magnitude_spectrum,
    mask,
    mo,
    plt,
    xlim,
    ylim,
):
    fig = plt.figure(figsize=(15, 5))

    val = crop_slider.value
    print(val)
    plt.subplot(221), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(224), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    plt.subplot(222), plt.imshow(img_back*val, cmap='gray')
    plt.title('Reconstructed Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(223), plt.imshow(imgc)
    plt.title('mask'), plt.xticks([]), plt.yticks([])
    print("orig: ",mask.shape, "new: ", (int(xlim*2), int(ylim*2)))

    mo.vstack([
    crop_slider,
    fig
              ])
    return


if __name__ == "__main__":
    app.run()
