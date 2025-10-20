import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


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

    from skimage import metrics
    import scipy
    return Generator, Tuple, cv2, metrics, np, os, plt, scipy


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
def _(images_from_directory):
    test_images = list(images_from_directory("Datasets/RIT_IREC_FULL/"))[::-1]
    return (test_images,)


@app.cell
def _(cv2, test_images):
    test_img = cv2.resize(test_images[8][1], (1280, 720), dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)

    return (test_img,)


@app.cell
def _(plt, test_img):
    plt.imshow(test_img)
    return


@app.cell
def _(cv2, test_img):
    ycbcr = cv2.cvtColor(test_img, cv2.COLOR_RGB2YCR_CB)

    return (ycbcr,)


@app.cell
def _(cv2, ycbcr):
    downsize_chrominance_factor = 6

    chrominance_shape = (ycbcr.shape[1]//downsize_chrominance_factor, ycbcr.shape[0]//downsize_chrominance_factor)
    Y = ycbcr[:,:,0]
    Cr = cv2.resize(ycbcr[:,:,1], chrominance_shape, dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)
    Cb = cv2.resize(ycbcr[:,:,2], chrominance_shape, dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)

    return Cb, Cr, Y


@app.cell
def _(Cb, Cr, Y, plt):
    plt.figure(figsize=(16,4)), plt.xticks([]), plt.yticks([])
    plt.title("Y Cb Cr (Downsampled)")
    plt.subplot(131), plt.imshow(Y, cmap="gray"), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(Cr, cmap="Reds"), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(Cb, cmap="Blues"), plt.xticks([]), plt.yticks([])

    plt.show()
    return


@app.cell
def _(Cb, Cr, Y):
    print(f"Y: {Y.min()} .. {Y.max()}")
    print(f"Cr: {Cr.min()} .. {Cr.max()}")
    print(f"Cb: {Cb.min()} .. {Cb.max()}")
    return


@app.cell
def _(Tuple, np):
    def crop_dct(dct_2d: np.ndarray, quality: float) -> (np.ndarray, Tuple[int, int]):
        # returns downsampled version, original size
        r, c = int(dct_2d.shape[0]*quality), int(dct_2d.shape[1]*quality)
        cropped_dct = dct_2d[0:r, 0:c]
        return cropped_dct, dct_2d.shape

    def uncrop_dct(cropped_dct: np.ndarray, original_size: Tuple[int, int]) -> (np.ndarray):
        shape = cropped_dct.shape
        og_dct = np.zeros(original_size)
        og_dct[0:shape[0] , 0:shape[1]] = cropped_dct
        return og_dct
    return crop_dct, uncrop_dct


@app.cell
def _():
    Yqual = .07
    Cqual = .025
    return Cqual, Yqual


@app.cell
def _(Cb, Cqual, Cr, Y, Yqual, crop_dct, scipy):
    Ydct, Ydct_og_size = crop_dct(scipy.fft.dctn((Y-128)/128.0), Yqual)
    Crdct, Crdct_og_size = crop_dct(scipy.fft.dctn((Cr-128)/128.0), Cqual)
    Cbdct, Cbdct_og_size = crop_dct(scipy.fft.dctn((Cb-128)/128.0), Cqual)
    return Cbdct, Cbdct_og_size, Crdct, Ydct, Ydct_og_size


@app.cell
def _(np):
    def visualize_dct(d):
        d = np.log(abs(d).clip(0.1))
        maxi, mini = d.max(), d.min()
        d = 255*(d - mini)/(maxi-mini)
        return d
    return (visualize_dct,)


@app.cell
def _(Cbdct, Crdct, Ydct, plt, visualize_dct):
    print(Crdct.shape, Crdct.dtype)


    plt.figure(figsize=(16,4)), plt.xticks([]), plt.yticks([])
    plt.title("Y Cb Cr (DCT Space)")
    plt.subplot(131), plt.imshow(visualize_dct(Ydct), cmap="gray"), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(visualize_dct(Crdct), cmap="Reds"), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(visualize_dct(Cbdct), cmap="Blues"), plt.xticks([]), plt.yticks([])

    plt.show()
    return


@app.cell
def _(
    Cb,
    Cbdct,
    Cbdct_og_size,
    Cr,
    Crdct,
    Y,
    Ydct,
    Ydct_og_size,
    cv2,
    plt,
    scipy,
    uncrop_dct,
):
    iY = (scipy.fft.idctn(uncrop_dct(Ydct, Ydct_og_size)))
    iCr = (scipy.fft.idctn(uncrop_dct(Crdct, Cbdct_og_size)))
    iCb = (scipy.fft.idctn(uncrop_dct(Cbdct, Cbdct_og_size)))

    print("uiy", iY.shape, Ydct_og_size, Cbdct_og_size)

    plt.figure(figsize=(16,8)), plt.xticks([]), plt.yticks([])
    plt.title("Y Cb Cr (Reconstruction with no cropping)")
    plt.subplot(231), plt.imshow(iY, cmap="gray"), plt.xticks([]), plt.yticks([])
    plt.subplot(232), plt.imshow(iCr, cmap="Reds"), plt.xticks([]), plt.yticks([])
    plt.subplot(233), plt.imshow(iCb, cmap="Blues"), plt.xticks([]), plt.yticks([])

    plt.subplot(234), plt.imshow(Y, cmap="gray"), plt.xticks([]), plt.yticks([])
    plt.subplot(235), plt.imshow(Cr, cmap="Reds"), plt.xticks([]), plt.yticks([])
    plt.subplot(236), plt.imshow(Cb, cmap="Blues"), plt.xticks([]), plt.yticks([])


    iCrb = cv2.resize(iCr, (1280, 720), dst=None, fx=None, fy=None, interpolation=cv2.INTER_NEAREST)
    iCbb = cv2.resize(iCb, (1280, 720), dst=None, fx=None, fy=None, interpolation=cv2.INTER_NEAREST)

    print(iCbb.shape)

    plt.show()

    return iCbb, iCrb, iY


@app.cell
def _(cv2, iCbb, iCrb, iY, np, plt, test_img):
    img_ycrcb = np.stack([iY, iCrb, iCbb], axis=-1)
    print(img_ycrcb.shape)
    img_rgb = cv2.cvtColor(img_ycrcb.astype(np.uint8), cv2.COLOR_YCR_CB2RGB)

    plt.figure(figsize=(16,4)), plt.xticks([]), plt.yticks([])
    plt.title("Before and After")
    plt.subplot(121), plt.imshow(test_img), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_rgb.astype(int)), plt.xticks([]), plt.yticks([])

    plt.show()
    return (img_rgb,)


@app.cell
def _(Cbdct, Crdct, Ydct):
    print(Ydct.min(), Ydct.max())
    total = Ydct.shape[0]*Ydct.shape[1]*4 + Crdct.shape[0] * Crdct.shape[1]*4 + Cbdct.shape[0] * Cbdct.shape[1]*4
    print(Ydct.shape[0]*Ydct.shape[1]*4, Crdct.shape[0] * Crdct.shape[1]*4, Cbdct.shape[0] * Cbdct.shape[1]*4, total) # 38016
    return


@app.cell
def _(test_img):
    print(f"{test_img.shape[0]}x{test_img.shape[1]} = {test_img.shape[0] * test_img.shape[1]} * RGB = {test_img.shape[0] * test_img.shape[1]*3}")
    return


@app.cell
def _(img_rgb, metrics, np, plt, test_img):
    print(np.square(test_img - img_rgb).mean())
    ssim = metrics.structural_similarity(test_img, img_rgb, full=True, data_range = img_rgb.max() - img_rgb.min(), channel_axis=2)
    print("SSIM: ", ssim[0])
    plt.imshow(np.abs(test_img - img_rgb)), plt.xticks([]), plt.yticks([])
    plt.show()
    return


if __name__ == "__main__":
    app.run()
