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
    It gets encoded into many Packets which are radio packets with a maximum length of 256 bytes
    """
    )
    return


@app.cell
def _():
    LORA_MAX_PACKET=256
    return (LORA_MAX_PACKET,)


@app.cell
def _(List, np):
    type Packet = bytearray

    class Format:
        def name() -> str:
            return "unnamed format"
        def encode(image: np.ndarray) -> List[Packet]:
            return []

        def decode(packets: List[Packet]) -> np.ndarray | None:
            return []
    return Format, Packet


@app.cell
def _(np):
    def mse_image(original: np.ndarray, compressed: np.ndarray):
    
        if original.shape != compressed.shape:
            raise ValueError("Differently sized images. Try resizing before hand")
    
        squared_diffs = np.square(original - compressed)
        return np.mean(squared_diffs)

    return


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
def _(BytesIO, Format, Image, LORA_MAX_PACKET, List, Packet, np):
    class JPEGEncoder(Format):
        def __init__(self, quality):
            self.quality = quality
        def name(self) -> str:
            return f"JPEG Encoder Quality {self.quality}"
        def encode(self, image: np.ndarray) -> List[Packet]:    
            byte_arr = BytesIO()
        
            img = Image.fromarray(image)
            img.save(byte_arr, format='JPEG', quality=self.quality)

            allbytes = byte_arr.getvalue()
            pacs = [allbytes[i:min(i+LORA_MAX_PACKET, len(allbytes))] for i in range(0, len(allbytes), LORA_MAX_PACKET)]
        
            return pacs
    
        def decode(self, packets: List[Packet]) -> np.ndarray | None:
            buf = bytearray()

            for pac in packets:
                buf.extend(pac)
        
            image = Image.open(BytesIO(buf))
        
            return np.array(image)
    return (JPEGEncoder,)


@app.cell
def _(images_from_directory):
    rit25 = list(images_from_directory('Datasets/RIT_IREC_FULL/'))
    return (rit25,)


@app.cell
def _(plt, rit25):
    plt.imshow(rit25[2][1])
    plt.show()
    return


@app.cell
def _(JPEGEncoder, rit25):
    enc10 = JPEGEncoder(10)
    pacs = enc10.encode(rit25[2][1])
    return enc10, pacs


@app.cell
def _(enc10, pacs):
    decoded = enc10.decode(pacs)
    return (decoded,)


@app.cell
def _(decoded, plt):
    plt.imshow(decoded)
    plt.show()
    return


@app.cell
def _(JPEGEncoder):
    encoders = [JPEGEncoder(1), JPEGEncoder(50), JPEGEncoder(100)]
    return (encoders,)


@app.cell
def _(encoders, plt, rit25):
    num_encoders = len(encoders)
    fig, axs = plt.subplots(num_encoders, 2, figsize=(9,10))
    for row, encoder in zip(axs, encoders):
        packets = encoder.encode(rit25[2][1])
        output = encoder.decode(packets)
        num_pacs = len(packets)
        num_bytes = sum(map(lambda p : len(p), packets))
        print(f"{encoder.name()}, Packets: {num_pacs}, Bytes: {num_bytes}")
        #reference 
        row[0].set_title("Reference")
        row[0].imshow(rit25[2][1])
        # decoded
        row[1].set_title(f"Result: {encoder.name()}")
        row[1].imshow(output)

    fig.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
