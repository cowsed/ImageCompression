import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    More things to consider
    - Run Length Encoding individual Packets
        - if shorter packets decode better, this could help there
    - For mmaking packets, rays from the origin
      - encode that signal, based on vis, seem compressible (pretty much just a downard slope with noise)


    - Tiled DCT
      - kinda like what jpeg does, do dcts on subchunks
      - kinda undoes some of our goals of loading whole image
      - but if few enough subchunks, could still be good
    """
    )
    return


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

    from skimage import metrics
    import scipy
    import pandas as pd
    import itertools
    import math
    import tempfile
    return (
        BytesIO,
        Generator,
        Image,
        List,
        Tuple,
        cv2,
        math,
        metrics,
        np,
        os,
        pd,
        plt,
        scipy,
        tempfile,
    )


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
        # return packets and byte estimate
        def encode(image: np.ndarray) -> (List[Packet], int):
            return []

        def decode(packets: List[Packet], size: Tuple[int, int]) -> np.ndarray | None:
            return []
    return Format, Packet


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Metrics
    need some way to determine if the image looks good

    ### MSE
    its like fine iguess, not how humans see it but whatevs
    """
    )
    return


@app.cell
def _(np):
    def mse_image(original: np.ndarray, compressed: np.ndarray):

        if original.shape != compressed.shape:
            raise ValueError(f"Differently sized images {original.shape} vs {compressed.shape}. Try resizing before hand")

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
def _(images_from_directory):
    rit25 = list(images_from_directory('Datasets/RIT_IREC_FULL/'))
    wenet = list(images_from_directory('Datasets/wenet_test_images//'))
    return rit25, wenet


@app.cell
def _(mo):
    mo.md(
        r"""
    # Encoders 
    they encode and decode
    ## Silly JPEG Encoder
    just encodes the image as a jpeg, splits it into 256 byte packets and sends them all
    """
    )
    return


@app.cell
def _(BytesIO, Format, Image, LORA_MAX_PACKET, List, Packet, Tuple, np):
    class JPEGEncoder(Format):
        def __init__(self, quality):
            self.quality = int(quality)
        def name(self) -> str:
            return f"JPEG Encoder Quality {self.quality}"
        def encode(self, image: np.ndarray) -> List[Packet]:    
            byte_arr = BytesIO()

            img = Image.fromarray(image)
            img.save(byte_arr, format='JPEG', quality=self.quality)
            allbytes = byte_arr.getvalue()
            pacs = [allbytes[i:min(i+LORA_MAX_PACKET, len(allbytes))] for i in range(0, len(allbytes), LORA_MAX_PACKET)]

            return pacs, len(allbytes)

        def decode(self, packets: List[Packet], size: Tuple[int, int]) -> np.ndarray | None:
            buf = bytearray()

            for pac in packets:
                buf.extend(pac)

            image = Image.open(BytesIO(buf))

            return np.array(image)
    return (JPEGEncoder,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Halfway Grayscale FFT Encoder
    FFT that stuff, crop it, send it (somehow), uncrop and fill with zeros, IFFT it
    """
    )
    return


@app.cell
def _(Format, List, Packet, Tuple, np):
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
            img = image

            f_transform = np.fft.fft2(img)
            f_transform_shifted_init = np.fft.fftshift(f_transform)
            f_transform_shifted = f_transform_shifted_init 

            ylim = int(size[0]/2 * self.value)
            xlim = int(size[1]/2 * self.value)

            yextent, xextent = int(ylim * 2), int(xlim * 2)

            y_start = int(f_transform_shifted.shape[0]/2 - ylim)
            x_start = int(f_transform_shifted.shape[1]/2 - xlim)

            masked = f_transform_shifted[y_start:y_start+yextent, x_start:x_start+xextent]
            mask_shape = masked.shape

            return masked

        def encode(self, image: np.ndarray) -> List[Packet]:
            size = image.shape
            if len(image.shape) > 2:
                raise ValueError("Only support one channel!")
            fdomain = self.encode_to_fdomain(image, size)
            return fdomain.reshape(-1), fdomain.shape[0] * fdomain.shape[1] * 2 * 4

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
def _(mo):
    mo.md(
        r"""
    ## Halfway Grayscale FFT Encoder
    DCT that stuff, crop it, send it (somehow), uncrop and fill with zeros, IDCT it
    """
    )
    return


@app.cell
def _(Format, List, Packet, Tuple, np, scipy):
    class GrayscaleDCTFloat(Format):
        def __init__(self, quality):
            self.quality = quality
            self.value = self.quality / 100
        def name(self):
            return f"Grayscale DCT Float Quality {self.quality}"
        def encode(self, image: np.ndarray) -> List[Packet]:
            size = image.shape
            if len(image.shape) > 2:
                raise ValueError("Only support one channel!")
            gray = image
            dct = scipy.fft.dctn(gray)
            mask_shape = (int( size[0] * self.value) ),  int( size[1] * self.value )
            masked = dct[0:mask_shape[0], 0:mask_shape[1]]

            return masked, masked.shape[0] * masked.shape[1] * 4

        def decode(self, packets: List[Packet], size: Tuple[int, int]) -> np.ndarray:
            d2 = packets
            mask_shape = (int( size[0] * self.value) ),  int( size[1] * self.value )
            dct = np.zeros(size)
            dct[0:mask_shape[0], 0:mask_shape[1]] = d2
            idct = scipy.fft.idctn(dct)
            return idct
    return (GrayscaleDCTFloat,)


@app.cell
def _(Format, np):
    class RGBFromGrayscale(Format):
        def __init__(self, grayscale_encoder, quality):
            self.quality = quality
            self.grayscale_encoder = grayscale_encoder(quality)
        def name(self):
            return f"RGB via 1D {self.grayscale_encoder.name()}"
        def encode(self, image: np.ndarray):
            if len(image.shape) != 3 or image.shape[2] !=3:
                raise ValueError("I require 3 channel color images")
            r = image[:,:,0]
            g = image[:,:,1]
            b = image[:,:,2]

            renc, s1 = self.grayscale_encoder.encode(r)
            genc, s2 = self.grayscale_encoder.encode(g)
            benc, s3 = self.grayscale_encoder.encode(b)

            return (renc, genc, benc), s1 + s2 + s3
        def decode(self, packets, size):
            renc,genc,benc = packets
            r = self.grayscale_encoder.decode(renc, size[0:2]).astype(int)
            g = self.grayscale_encoder.decode(genc, size[0:2]).astype(int)
            b = self.grayscale_encoder.decode(benc, size[0:2]).astype(int)
            combined = np.stack([r,g,b], axis=-1)

            return combined
    return (RGBFromGrayscale,)


@app.cell
def _(Format, cv2, np):
    class YCbCrFromGrayscale(Format):
        def __init__(self, one_channel_encoder, Yquality, CbCrQuality):
            self.Yquality = Yquality
            self.CbCrQuality = CbCrQuality
            self.brightness_encoder = one_channel_encoder(Yquality)
            self.cbcr_encoder = one_channel_encoder(CbCrQuality)
        def name(self):
            return f"YCrYCb via Y {self.brightness_encoder.name()} and CrCb {self.cbcr_encoder.name()}"
        def encode(self, image_rgb: np.ndarray):
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCR_CB)

            if len(image.shape) != 3 or image.shape[2] !=3:
                raise ValueError("I require 3 channel color images")
            y = image[:,:,0]
            cr = image[:,:,1]
            cb = image[:,:,2]

            yenc, s1 = self.brightness_encoder.encode(y)
            crenc, s2 = self.cbcr_encoder.encode(cr)
            cbenc, s3 = self.cbcr_encoder.encode(cb)

            return (yenc- 128, crenc- 128, cbenc- 128), s1 + s2 + s3

        def decode(self, packets, size):
            yenc, crenc, cbenc = packets

            y = self.brightness_encoder.decode(yenc+128, size[0:2])
            cr = self.cbcr_encoder.decode(crenc+128, size[0:2])
            cb = self.cbcr_encoder.decode(cbenc+128, size[0:2])

            combined = np.stack([y,cr, cb], axis=-1)
            combined_rgb = cv2.cvtColor(combined.astype(np.uint8), cv2.COLOR_YCR_CB2RGB).astype(int)
            return np.clip(combined_rgb, 0, 255)
    return (YCbCrFromGrayscale,)


@app.cell
def _(Format, Tuple, cv2, np, scipy):
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

    class KnockOffGJpegButNotTiledAndNotCenteredAtZero(Format):
        def __init__(self, Yqual, Cqual, Cdivision):
            self.Yqual = Yqual/100
            self.Cqual = Cqual/100 * Cdivision
            self.CDivision = Cdivision
        def name(self) -> str:
            return f"KnockOffJpeg: YQ: {self.Yqual:.2}, CQ: {self.Cqual:.2}, CDiv: {self.CDivision}"
        def encode(self, image_rgb: np.ndarray):
            ycbcr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCR_CB)

            chrominance_shape = (ycbcr.shape[1]//self.CDivision, ycbcr.shape[0]//self.CDivision)
            Y = ycbcr[:,:,0]
            Cr = cv2.resize(ycbcr[:,:,1], chrominance_shape, dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)
            Cb = cv2.resize(ycbcr[:,:,2], chrominance_shape, dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)

            Ydct, Ydct_og_size = crop_dct(scipy.fft.dctn(Y), self.Yqual)
            Crdct, Crdct_og_size = crop_dct(scipy.fft.dctn(Cr), self.Cqual)
            Cbdct, Cbdct_og_size = crop_dct(scipy.fft.dctn(Cb), self.Cqual)
            num_floats = (Ydct.shape[0] * Ydct.shape[1] + Crdct.shape[0] * Crdct.shape[1] + Cbdct.shape[0] * Cbdct.shape[1])
            return (Ydct, Crdct, Cbdct), num_floats * 4

        def decode(self, packets, og_size):
            Ydct, Crdct, Cbdct = packets
            cr_og_size = (og_size[0]//self.CDivision, og_size[1]//self.CDivision)

            iY =  scipy.fft.idctn(uncrop_dct(Ydct, og_size[:2]))
            iCr = scipy.fft.idctn(uncrop_dct(Crdct, cr_og_size))
            iCb = scipy.fft.idctn(uncrop_dct(Cbdct, cr_og_size))

            iCrb = cv2.resize(iCr, og_size[:2][::-1], dst=None, fx=None, fy=None, interpolation=cv2.INTER_NEAREST)
            iCbb = cv2.resize(iCb, og_size[:2][::-1], dst=None, fx=None, fy=None, interpolation=cv2.INTER_NEAREST)

            img_ycrcb = np.stack([iY, iCrb, iCbb], axis=-1)

            img_rgb = cv2.cvtColor(np.clip(img_ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCR_CB2RGB)
            return img_rgb



    class KnockOffGJpegButNotTiledFloat(Format):
        def __init__(self, Yqual, Cqual, Cdivision):
            self.Yqual = Yqual/100
            self.Cqual = Cqual/100 * Cdivision
            self.CDivision = Cdivision
        def name(self) -> str:
            return f"KnockOffJpeg: YQ: {self.Yqual:.2}, CQ: {self.Cqual:.2}, CDiv: {self.CDivision}"
        def encode(self, image_rgb: np.ndarray):
            img_float = (image_rgb/255.0 - .5).astype(np.float32)
            ycbcr = cv2.cvtColor(img_float, cv2.COLOR_RGB2YCR_CB)

            chrominance_shape = (ycbcr.shape[1]//self.CDivision, ycbcr.shape[0]//self.CDivision)
            Y = ycbcr[:,:,0]
            Cr = cv2.resize(ycbcr[:,:,1], chrominance_shape, dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)
            Cb = cv2.resize(ycbcr[:,:,2], chrominance_shape, dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)

            Ydct, Ydct_og_size = crop_dct(scipy.fft.dctn(Y), self.Yqual)
            Crdct, Crdct_og_size = crop_dct(scipy.fft.dctn(Cr), self.Cqual)
            Cbdct, Cbdct_og_size = crop_dct(scipy.fft.dctn(Cb), self.Cqual)
            num_floats = (Ydct.shape[0] * Ydct.shape[1] + Crdct.shape[0] * Crdct.shape[1] + Cbdct.shape[0] * Cbdct.shape[1])
            return (Ydct, Crdct, Cbdct), num_floats * 4

        def decode(self, packets, og_size):
            Ydct, Crdct, Cbdct = packets
            cr_og_size = (og_size[0]//self.CDivision, og_size[1]//self.CDivision)

            iY =  scipy.fft.idctn(uncrop_dct(Ydct, og_size[:2]))
            iCr = scipy.fft.idctn(uncrop_dct(Crdct, cr_og_size))
            iCb = scipy.fft.idctn(uncrop_dct(Cbdct, cr_og_size))

            iCrb = cv2.resize(iCr, og_size[:2][::-1], dst=None, fx=None, fy=None, interpolation=cv2.INTER_NEAREST)
            iCbb = cv2.resize(iCb, og_size[:2][::-1], dst=None, fx=None, fy=None, interpolation=cv2.INTER_NEAREST)

            img_ycrcb = np.stack([iY, iCrb, iCbb], axis=-1)

            img_rgb = cv2.cvtColor(img_ycrcb.astype(np.float32), cv2.COLOR_YCR_CB2RGB)+.5
        
            return np.clip(img_rgb*255, 0, 255).astype(np.uint8)
    return (
        KnockOffGJpegButNotTiledAndNotCenteredAtZero,
        KnockOffGJpegButNotTiledFloat,
    )


@app.cell
def _(List, Packet, Tuple, tempfile):
    def to_packets_via_io_files(func_from_to_bin, file_extension, max_packet_size) -> Tuple[List[Packet], int]:
        fname = tempfile.mktemp()+file_extension
        bin_fname = tempfile.mktemp()
        func_from_to_bin(fname, bin_fname)

        pacs = []
        size = 0
        with open(bin_fname, 'rb') as f:
            while True:
                chunk = f.read(max_packet_size)
                if not chunk: 
                    break
                size+=len(chunk)
                pacs.append(chunk)
        return pacs, size

    def to_image_from_packets(convert_fn, packets):
        bin_fname = tempfile.mktemp()
        fname = tempfile.mktemp()
        with open(bin_fname, 'wb') as f:
            for packet in packets:
                f.write(packet)
        return convert_fn(bin_fname, fname)
    return to_image_from_packets, to_packets_via_io_files


@app.cell
def _(
    Format,
    List,
    Packet,
    cv2,
    os,
    to_image_from_packets,
    to_packets_via_io_files,
):
    class SSDVEncoder(Format):
        def __init__(self, quality, max_packet_size, callsign, image_id):
            if (max_packet_size > 256):
                raise ValueError("Packet size too large (max 256)")
            self.qual = quality
            self.max_packet_size = max_packet_size
            self.callsign = callsign
            self.image_id = image_id % 256
        def name(self):
            return f"SSDV Encoder: Quality {self.qual}"
        def encode(self, image_rgb) -> List[Packet]:
            if image_rgb.shape[0] % 16 != 0 or image_rgb.shape[1] % 16 != 0:
                raise ValueError("Dimensions must be multiples of 16")

            def convert(fname, bin_fname):
                cv2.imwrite(fname, image_rgb)
                cmd = f"ssdv -e -n -q {self.qual} -c {self.callsign} -i {self.image_id} -l {self.max_packet_size} {fname} {bin_fname}"
                ret = os.system(cmd)
                if ret != 0:
                    raise RuntimeError(f"Couldnt encode with SSDV: {ret}")
            return to_packets_via_io_files(convert, ".jpg", self.max_packet_size)

        def decode(self, packets, og_size):
            def convert(bin_fname, fname):
                cmd = f"ssdv -d -l {self.max_packet_size} {bin_fname} {fname} 2>/dev/null"
                ret = os.system(cmd)
                if ret != 0:
                    raise RuntimeError("Couldnt decode with SSDV")
                img = cv2.imread(fname)
                return img

            return to_image_from_packets(convert, packets)
    return (SSDVEncoder,)


@app.cell
def _(
    Format,
    List,
    Packet,
    cv2,
    os,
    to_image_from_packets,
    to_packets_via_io_files,
):
    class JPEG2000EncoderNoPackets(Format):
        def __init__(self, quality, max_packet_size):
            self.qual = int(quality)
            self.max_packet_size = max_packet_size
        def name(self):
            return f"JPEG2000 Encoder: Quality {self.qual}"
        def encode(self, image_rgb) -> List[Packet]:

            def convert(fname, bin_fname):
                cv2.imwrite(fname, image_rgb)
                cmd = f"convert {fname} -quality {self.qual} {bin_fname}.jp2 2>/dev/null && mv {bin_fname}.jp2 {bin_fname}"
                ret = os.system(cmd)
                if ret != 0:
                    raise RuntimeError("Couldnt encode with JPEG2000")
            return to_packets_via_io_files(convert, ".png", self.max_packet_size)

        def decode(self, packets, og_size):
            def convert(bin_fname, fname):
                cmd = f"mv {bin_fname} {bin_fname}.jp2 && convert {bin_fname}.jp2 {fname}.jpg 2>/dev/null"
                ret = os.system(cmd)
                if ret != 0:
                    raise RuntimeError("Couldnt decode with JPEG2000")

                img = cv2.imread(f"{fname}.jpg")
                return img

            return to_image_from_packets(convert, packets)
    return (JPEG2000EncoderNoPackets,)


@app.cell
def _(mo):
    mo.md(r"""# Lora Airtime Calculator""")
    return


@app.cell
def _(math, namedtuple):
    LoraSettings = namedtuple("LoraSettings", ["SF", "BW", "CR", "PreambleSize", "CrcOn", "ImplicitHeader", "LDR", "MaxSize"])
    # Source: https://github.com/ifTNT/lora-air-time

    def n_symbol(payload_len: int, settings: LoraSettings):
        payload_bits = 8 * payload_len
        payload_bits -= settings.SF * 4
        payload_bits += 8
        if settings.CrcOn:
            payload_bits+=16

        if settings.ImplicitHeader:
            payload_bits+=16
        payload_bits = max(payload_bits, 0)

        bits_per_symbol = settings.SF
        if settings.LDR:
            bits_per_symbol -= 2
        payload_symbol = math.ceil(payload_bits / 4 / bits_per_symbol) * settings.CR;
        payload_symbol += 8;

        preamble_symbols = settings.PreambleSize+4.25 
        return payload_symbol, preamble_symbols 

    def airtimeOne(payload_len: int, settings: LoraSettings) -> float:
        if payload_len > 255:
            raise ValueError("Max packet size = 255")
        T_s = (2**settings.SF)/settings.BW
        n_sym, n_pre = n_symbol(payload_len, settings)

        return T_s * (n_sym + n_pre) / 1000
    def airtime(payload_len: int, settings: LoraSettings) -> float:
        repeated = payload_len // settings.MaxSize
        remainder = payload_len % settings.MaxSize
        return repeated * airtimeOne(settings.MaxSize, settings) + airtimeOne(remainder, settings)
    return LoraSettings, airtime


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""# Playground""")
    return


@app.cell
def _(
    GrayscaleDCTFloat,
    GrayscaleFFTFloatEncoder,
    ICEREncoder,
    JPEG2000EncoderNoPackets,
    JPEGEncoder,
    KnockOffGJpegButNotTiledAndNotCenteredAtZero,
    KnockOffGJpegButNotTiledFloat,
    RGBFromGrayscale,
    SSDVEncoder,
    YCbCrFromGrayscale,
    mo,
    rit25,
    wenet,
):
    col_encoders = {
                "JPEG": lambda yqual, _, _2 : JPEGEncoder(yqual), 
                "JPEG2000": lambda yqual, _, _2 : JPEG2000EncoderNoPackets(int(yqual**2), 255), 
                "SSDV": lambda yqual, _, _2 : SSDVEncoder(int(yqual * 0.07), 64, 'KC1TPR', 1),
                "ICER": lambda yqual, _, _2 : ICEREncoder(yqual * 1000),
                "RGBFFT": lambda yqual, chrqual, _ : RGBFromGrayscale(GrayscaleFFTFloatEncoder, yqual), 
                "RGBDCT": lambda yqual, chrqual,_ : RGBFromGrayscale(GrayscaleDCTFloat, yqual),
                "YCRCBFFT": lambda yqual, chrqual,_ : YCbCrFromGrayscale(GrayscaleFFTFloatEncoder, yqual, chrqual), 
                "YCRCBDCT": lambda yqual, chrqual,_ : YCbCrFromGrayscale(GrayscaleDCTFloat, yqual, chrqual),
                "NonTileJPEG": lambda yqual, chrqual, subdiv : KnockOffGJpegButNotTiledAndNotCenteredAtZero(yqual, chrqual, subdiv),
                "NonTileJPEGCentered": lambda yqual, chrqual, subdiv : KnockOffGJpegButNotTiledFloat(yqual, chrqual, subdiv)
               }



    col_fft_quality_slider = mo.ui.number(start=1, stop=99.9, step=0.1, label="Quality")
    col_chr_quality_slider = mo.ui.number(value = .2, start=.10, stop=99.9, step=0.1, label="Chrominance Quality")
    col_chr_subdiv_slider = mo.ui.number(value=4, start=1, stop=400, step=1, label="Chrominance Subdiv")
    col_image_selector = mo.ui.dropdown(options = {e[0]: e[1] for e in rit25 + wenet}, value=rit25[0][0], label = "Image")
    col_enc_selector = mo.ui.multiselect(options = col_encoders, value=["JPEG", "JPEG2000", "NonTileJPEGCentered"], label = "Encoders")
    return (
        col_chr_quality_slider,
        col_chr_subdiv_slider,
        col_enc_selector,
        col_fft_quality_slider,
        col_image_selector,
    )


@app.cell
def _(
    LoraSettings,
    airtime,
    col_chr_quality_slider,
    col_chr_subdiv_slider,
    col_enc_selector,
    col_fft_quality_slider,
    col_image_selector,
    cv2,
    metrics,
    mo,
    mse_image,
    np,
    plt,
):
    fig4 = plt.figure(figsize=(15, 8))

    col_img = cv2.resize(col_image_selector.value, (1280, 720), dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)


    def col_encoder_column(enc):
        middle, size = enc.encode(col_img)
        img_back = enc.decode(middle, col_img.shape)


        plt.subplot(312), plt.imshow(col_img)
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])

        plt.subplot(311), plt.imshow(img_back)
        plt.title('Reconstructed Image'), plt.xticks([]), plt.yticks([])

        plt.subplot(313), plt.imshow(np.abs(col_img - img_back))
        plt.title('Difference'), plt.xticks([]), plt.yticks([])

        mse = mse_image(col_img, img_back)
        ssim = metrics.structural_similarity(col_img, img_back, full=True, data_range = img_back.max() - img_back.min(), channel_axis=2)

        return mo.vstack([        
            mo.md(enc.name()), 
            mo.md(f"MSE (inf to 0 best): {mse:.4}"),
            mo.md(f"SSIM (-1 to 1 best): {ssim[0]:.4}"),
            mo.md(f"Bytes: {size}"),
            mo.md(f"Airtime (s): {airtime(size, LoraSettings(7, 125, 6, 8, True, False, False, 255))}"),
            fig4
        ])

    mo.vstack([
        mo.hstack([col_fft_quality_slider, col_chr_quality_slider, col_chr_subdiv_slider, col_enc_selector, col_image_selector]),
        mo.hstack(map(
            lambda etype : col_encoder_column(
                etype(float(col_fft_quality_slider.value), float(col_chr_quality_slider.value), col_chr_subdiv_slider.value)
            ), col_enc_selector.value))
    ])
    return


@app.cell
def _(np):
    def visualize_dct(d):
        d = np.log(abs(d).clip(0.1))
        maxi, mini = d.max(), d.min()
        d = 255*(d - mini)/(maxi-mini)
        return d
    return


@app.cell
def _(mo):
    val_slider = mo.ui.slider(start = 1, stop = 99.99, label = "quality")
    return


@app.cell
def _():
    from collections import namedtuple
    return (namedtuple,)


@app.cell
def _(metrics, mse_image, namedtuple):
    TestResult = namedtuple("TestResult", ["image", "bytes", "mse", "ssim", "psnr"])
    def test_encoder(encoder, original_image) -> TestResult:
        compressed_image, num_bytes = encoder.encode(original_image)
        decoded = encoder.decode(compressed_image, original_image.shape)
        mse = mse_image(original_image, decoded)
        ssim = metrics.structural_similarity(original_image, decoded, full=True, data_range = decoded.max() - decoded.min(), channel_axis=2)[0]
        psnr = metrics.peak_signal_noise_ratio(original_image, decoded, data_range = decoded.max() - decoded.min())


        return TestResult(decoded, num_bytes, mse, ssim, psnr)
    return (test_encoder,)


@app.cell
def _(
    JPEG2000EncoderNoPackets,
    JPEGEncoder,
    KnockOffGJpegButNotTiledAndNotCenteredAtZero,
    SSDVEncoder,
    cv2,
    rit25,
    test_encoder,
    wenet,
):
    # Yqualities = [.5, 1, 2, 4, 8, 16, 32, 64, 100][1:4]
    # Cqualities = [.5, 1, 2, 4, 8, 16, 32, 64, 100][1:4]
    # Subdivisions = [1, 2, 4, 8, 16, 32][1:3]
    results = []

    dataset=rit25[:5] + wenet[:5]
    jpeg_tests = [(image_name, image, "JPEG", JPEGEncoder, (int((1.3**qual)/2),)) for (image_name, image) in dataset for qual in range(6, 21, 2)]

    jpeg2000_tests = [(image_name, image, "JPEG2000", JPEG2000EncoderNoPackets, (int((1.25**qual)/2),255)) for (image_name, image) in dataset for qual in range(6, 21, 1)]


    fake_jpeg_tests = [(image_name, image, "YChrDctNonTile", KnockOffGJpegButNotTiledAndNotCenteredAtZero, (int((1.2**qual)/4),int((1.2**qual)/4), 8)) for (image_name, image) in dataset for qual in range(7, 26, 2)]

    ssdv_tests = [(image_name, image, "SSDV", lambda qual : SSDVEncoder(qual, 255, "KC1TPR", 0), (qual,)) for (image_name, image) in dataset for qual in [0, 15, 29, 43, 58, 72, 86, 100]]

    all_tests = jpeg_tests + jpeg2000_tests + fake_jpeg_tests #+ ssdv_tests

    for (image_name, image, encoder_id, encoder_factory, encoder_args) in all_tests:
        print(image_name, encoder_id, encoder_args)
        base_image = cv2.resize(image, (1280, 720), dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)
        encoder = encoder_factory(*encoder_args)
        result = test_encoder(encoder, base_image)
        results.append((image_name, result.image, encoder_id, encoder_args, encoder.name(), result.bytes, result.mse, result.ssim, result.psnr))
    return dataset, results


@app.cell
def _(pd, results):
    df = pd.DataFrame(results, columns=["image_name", "result_image", "encoder_type", "encoder_args", "encoder_name", "num_bytes", "mse", "ssim", "psnr"])
    # df["airtime"] = 
    return (df,)


@app.cell
def _(df):
    average_df = df[['encoder_type', 'encoder_args', 'encoder_name', 'num_bytes', 'mse', 'ssim', 'psnr']].groupby(['encoder_type', 'encoder_args', 'encoder_name']).mean()
    average_df
    return (average_df,)


@app.cell
def _(average_df, plt):
    fig, axs = plt.subplots(3, figsize=(8,8))

    for n, grp in average_df.reset_index().groupby('encoder_type'):
        axs[0].plot(grp['num_bytes'], grp["ssim"], marker='o', label=n)
        axs[1].plot(grp['num_bytes'], grp["mse"], marker='+', label=n)
        axs[2].plot(grp['num_bytes'], grp["psnr"], marker='+', label=n)

    axs[0].legend(title="Encoders")
    axs[1].legend(title="Encoders")
    axs[2].legend(title="Encoders")

    axs[0].set_xlabel("num bytes")
    axs[1].set_xlabel("num bytes")
    axs[2].set_xlabel("num bytes")

    axs[0].set_ylabel("SSIM Score (1 good, -1 bad)")
    axs[1].set_ylabel("MSE (higher is worse)")
    axs[2].set_ylabel("PSNR (higher is better)")

    axs[0].sharex(axs[1])
    axs[1].sharex(axs[2])
    axs[0].set_xscale('log')

    axs[2].set_xscale('log')

    plt.show()
    return


@app.cell
def _(dataset, mo):
    images_by_name = {e[0]:e[1] for e in dataset}
    result_image_picker = mo.ui.dropdown(options=images_by_name, value=dataset[0][0])
    return


@app.cell
def _(df, mo):
    encoder_results = {f"{row.encoder_type} - {row.encoder_args}": ind for ind, row in df.iterrows()}
    encoder_picker = mo.ui.multiselect(options=encoder_results, value=[])
    return


if __name__ == "__main__":
    app.run()
