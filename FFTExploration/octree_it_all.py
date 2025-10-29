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
    return Generator, Tuple, cv2, np, os, plt


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
def _(plt, test_images):
    plt.imshow(test_images[0][1])
    return


@app.cell
def _(cv2, np, test_images):
    test_img = cv2.resize(test_images[0][1], (256, 256), dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    return (test_img,)


@app.cell
def _(cv2, test_img):
    ycbcr = cv2.cvtColor(test_img, cv2.COLOR_RGB2YCR_CB)#[0:720, 450:450+720]

    return (ycbcr,)


@app.cell
def _(plt, ycbcr):
    plt.imshow(ycbcr[:,:,0])
    return


@app.cell
def _(np, ycbcr):
    Y = ycbcr[:,:,0].astype(np.int16)
    g0 = Y.mean()
    g0p = Y - g0

    return Y, g0, g0p


@app.cell
def _(g0, g0p, np, plt, ycbcr):
    plt.subplot(131), plt.imshow(np.full(ycbcr.shape[:2], g0), vmin=-255, vmax=255)
    plt.subplot(132), plt.imshow(np.full(ycbcr.shape[:2], g0p + 128), vmin=-255, vmax=255)
    plt.subplot(133), plt.imshow(np.full(ycbcr.shape[:2], g0p + g0), vmin=-255, vmax=255)

    return


@app.cell
def _(np):
    def subdiv_img(img):
        rs = img.shape[0]//2
        cs = img.shape[1]//2
        tl = img[0:rs,0:cs]
        tr = img[0:rs,cs:]
        bl = img[rs:,0:cs]
        br = img[rs:,cs:]
        return tl, tr, bl, br
    def remake_image(tl, tr, bl, br):
        assert tl.shape == tr.shape == bl.shape == br.shape
        rs = tl.shape[0]
        cs = tl.shape[1]
        fr = rs * 2
        fc = cs * 2
        img = np.zeros((fr, fc))
        img[0:rs,0:cs] = tl
        img[0:rs,cs:] = tr
        img[rs:,0:cs] = bl
        img[rs:,cs:] = br
        return img
    
    return remake_image, subdiv_img


@app.cell
def _(Tuple, np):
    def remove_mean(img) -> Tuple[np.int16, np.ndarray]:
        m = img.mean()
        return m, img - m
    def restore_mean(mean, img) -> np.ndarray:
        return mean + img
    return (remove_mean,)


@app.cell
def _(Y, plt, remake_image, subdiv_img):
    tl0, tr0, bl0, br0 = subdiv_img(Y)

    plt.subplot(221), plt.imshow(tl0, vmin=-255, vmax=255)
    plt.subplot(222), plt.imshow(tr0, vmin=-255, vmax=255)
    plt.subplot(223), plt.imshow(bl0, vmin=-255, vmax=255)
    plt.subplot(224), plt.imshow(br0, vmin=-255, vmax=255)
    plt.show()

    (remake_image(tl0, tr0, bl0, br0) == Y).all()

    return


@app.cell
def _(remove_mean, subdiv_img):
    def octree(level, img):
        if img.shape[0] == 0 or img.shape[0] == 0:
            raise ValueError("went too far, out of pixels")
        if level == 0:
            return img.mean()
        level -= 1
        m, next = remove_mean(img)
        parts = list(subdiv_img(next))
        subos = tuple(map(lambda i : octree(level, i), parts))
    
        return m, subos
    return (octree,)


@app.cell
def _(np, remake_image):
    def deoctree(m, subs, full_shape, ttl):
        half_shape = (full_shape[0]//2, full_shape[1]//2)
        if not isinstance(subs[0], tuple):
            return m+remake_image(*list(map(lambda me : np.full(half_shape, me), subs)))
    
        # if ttl < 1:
        #     tl, tr, bl, br = tuple(map(lambda a : np.full(half_shape, a[0]), subs))
        #     return remake_image(tl, tr, bl, br)        
        tl, tr, bl, br = tuple(map(lambda a : deoctree(a[0], a[1], half_shape, ttl - 1), subs))
        return remake_image(tl, tr, bl, br) + m
    return (deoctree,)


@app.cell
def _(Y, deoctree, octree, plt):
    m, subs = octree(8, Y)
    rimg = deoctree(m, subs, Y.shape, 10)
    plt.subplot(121), plt.imshow(Y, vmin=-255, vmax=255, cmap="gray")
    plt.subplot(122), plt.imshow(rimg, vmin=-255, vmax=255, cmap="gray")
    plt.show()
    (rimg == Y).all()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
