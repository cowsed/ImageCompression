import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from typing import List, Dict, Generator, Tuple
    import numpy as np
    import os
    import cv2
    import pathlib
    import io

    from PIL import Image
    from io import BytesIO
    import base64
    import matplotlib.pyplot as plt
    return BytesIO, Generator, Image, List, Tuple, cv2, np, os, plt


@app.cell
def _(mo):
    mo.md(
        r"""
    # Type definitions for automatic testing

    An Image is an array of RGB values
    It gets encoded into many Packets which are radio packets with a maximum length of 244 bytes
    """
    )
    return


@app.cell
def _():
    LORA_MAX_PACKET=255
    return (LORA_MAX_PACKET,)


@app.cell
def _(List, Tuple, np):
    type Packet = bytearray

    class Format:
        def name() -> str:
            return "unnamed format"
        def encode(image: np.ndarray, size: Tuple(int, int)) -> List[Packet]:
            return []

        def decode(packets: List[Packet], size: Tuple(int, int)) -> np.ndarray | None:
            return []
    return Format, Packet


@app.cell
def _():
    return


@app.cell
def _(np):
    def mse_image(original: np.ndarray, compressed: np.ndarray):

        if original.shape != compressed.shape:
            raise ValueError("Differently sized images. Try resizing before hand")

        squared_diffs = np.square(original - compressed)
        return np.mean(squared_diffs)
    return (mse_image,)


@app.cell
def _(Generator, Tuple, cv2, np, os):
    def images_from_directory(dir: str)-> Generator[Tuple[str,np.ndarray]]:
        listing = os.listdir(dir)
        print(listing)
        for item in listing:
            path = os.path.join(dir, item)

            img = cv2.imread(path, 3)
            if img is None:
                print(f"Could not load {item} as image")
                continue
            yield item, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (images_from_directory,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Silly JPEG Encoder
    just encodes the image as a jpeg, splits it into 256 byte packets and sends them all
    """
    )
    return


@app.cell
def _(images_from_directory):
    rit25 = list(images_from_directory('Datasets/RIT_IREC_FULL/'))
    return (rit25,)


@app.cell
def _(BytesIO, Format, Image, LORA_MAX_PACKET, List, Packet, Tuple, np):
    class JPEGEncoder(Format):
        def __init__(self, quality):
            self.quality = quality
        def name(self) -> str:
            return f"JPEG Encoder Quality {self.quality}"
        def encode(self, image: np.ndarray, size: Tuple[int, int]) -> List[Packet]:    
            byte_arr = BytesIO()

            img = Image.fromarray(image)
            img.save(byte_arr, format='JPEG', quality=self.quality)

            allbytes = byte_arr.getvalue()
            pacs = [allbytes[i:min(i+LORA_MAX_PACKET, len(allbytes))] for i in range(0, len(allbytes), LORA_MAX_PACKET)]

            return pacs

        def decode(self, packets: List[Packet], size: Tuple[int, int]) -> np.ndarray | None:
            buf = bytearray()

            for pac in packets:
                buf.extend(pac)

            image = Image.open(BytesIO(buf))

            return np.array(image)
    return


@app.cell
def _(Format, List, Packet, Tuple, cv2, np):
    class GrayscaleFFTFloatEncoder(Format):
        # quality 1-100
        def __init__(self, quality):
            if (quality < 1 or quality >= 100):
                raise ValueError("1 <= Quality < 100")
            self.quality = quality
            self.value = quality / 100

        def name(self):
            return f"Grayscale FFT Encoder Quality {self.quality}"

        def encode_to_fdomain(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            f_transform = np.fft.fft2(img)
            f_transform_shifted_init = np.fft.fftshift(f_transform)
            f_transform_shifted = f_transform_shifted_init 

            ylim = int(size[0]/2 * self.value)
            xlim = int(size[1]/2 * self.value)
        
            yextent, xextent = int(ylim * 2), int(xlim * 2)

            y_start = int(f_transform_shifted.shape[0]/2 - ylim)
            x_start = int(f_transform_shifted.shape[1]/2 - xlim)

            masked = f_transform_shifted[y_start:y_start+yextent, x_start:x_start+xextent]

            return masked;

        def encode(self, image: np.ndarray, size: Tuple[int, int]) -> List[Packet]:
            fdomain = self.encode_to_fdomain(image, size)
            return fdomain.reshape(-1)
    
        def decode_from_fdomain(self, fdomain: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
            ylim = int(size[0]/2 * self.value)
            xlim = int(size[1]/2 * self.value)

            y_start = int(size[0]/2 - ylim)
            x_start = int(size[1]/2 - xlim)
            yextent, xextent = int(ylim * 2), int(xlim * 2)

        
            f_transform = np.zeros(size[0:2], dtype=np.complex128)
            f_transform[y_start:y_start+yextent, x_start:x_start+xextent] = fdomain
        
            f_transform_back = f_transform
        

            # Perform 2D Inverse FFT
            # np.fft.ifftshift shifts the zero-frequency component back to the top-left
            # np.fft.ifft2 computes the 2D inverse discrete Fourier Transform
            # np.abs() is used to get the real part of the image, as IFFT can return complex numbers
            f_ishift = np.fft.ifftshift(f_transform_back)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)

            return img_back
        def decode(self, packets: List[Packet], size: Tuple[int, int]) -> np.ndarray:
            d1 = packets
            ylim = int(size[0]/2 * self.value)
            xlim = int(size[1]/2 * self.value)
        
            yextent, xextent = int(ylim * 2), int(xlim * 2)
            fdomain_shape = (yextent, xextent)

            fdomain = d1.reshape(fdomain_shape)
        
            return self.decode_from_fdomain(fdomain, size)
        
        
    return (GrayscaleFFTFloatEncoder,)


@app.cell
def _():
    return


@app.cell
def _(mo, rit25):
    fft_quality_slider = mo.ui.slider(start=1, stop=99.9, step=0.1, label="Quality")
    image_selector = mo.ui.dropdown(options = {e[0]: e[1] for e in rit25}, value=rit25[0][0], label = "Image")
    return fft_quality_slider, image_selector


@app.cell
def _(
    GrayscaleFFTFloatEncoder,
    cv2,
    fft_quality_slider,
    image_selector,
    mo,
    mse_image,
    plt,
):
    fig3 = plt.figure(figsize=(15, 8))

    img = cv2.resize(image_selector.value, (1200, 800), dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)

    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    val = fft_quality_slider.value

    enc = GrayscaleFFTFloatEncoder(val)
    middle = enc.encode(img, img.shape)
    img_back = enc.decode(middle, img.shape)
    plt.subplot(131), plt.imshow(gimg, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(132), plt.imshow(img_back, cmap='gray')
    plt.title('Reconstructed Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(133), plt.imshow(gimg - img_back, cmap='gray')
    plt.title('Difference'), plt.xticks([]), plt.yticks([])

    print(f"Compression Quality: {val}, MSE: {mse_image(gimg, img_back)}")

    mo.vstack([
        mo.hstack([fft_quality_slider, image_selector]),
        fig3
    ])


    return


@app.cell
def _(GrayscaleFFTFloatEncoder, cv2, mse_image, plt, rit25):
    genc = GrayscaleFFTFloatEncoder(15)

    for entry in rit25[::1]:
        test_img = test_img = cv2.resize(entry[1], (1200, 800), dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)

        gtest_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        enced = genc.encode(test_img, gtest_img.shape)
        deced = genc.decode(enced, gtest_img.shape)
    
        mse = mse_image(gtest_img, deced)
        print(f"{entry[0]} Dims: {gtest_img.shape}, MSE: {mse}")
    
        diff = gtest_img - deced
        fig2 = plt.figure(figsize=(14,16))
        plt.subplot(131), plt.imshow(gtest_img, cmap="gray")
        plt.subplot(132), plt.imshow(deced, cmap="grey")
        plt.subplot(133), plt.imshow(diff, cmap="grey")
        plt.show()
    return


if __name__ == "__main__":
    app.run()
