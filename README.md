# Image Compression Experiments for IREC 2026 Payload

## Goals
- Provide progressively loading
  - If we took a picture of the inside of the payload, don't wait 15 minutes for it all to get here
- Resillience to Packet Drops
  - The LoRa radio link can provide FEC and Checksums which increase our ability to decode a packet and drops packets which are not fully decoded
- High Compression Ratio
  - Packets can take a while to send, and can only hold max 255 bytes (but less could be better) we need to compress the images as much as we can
 

## Layout

So far, the bulk of this project is in FFTExploration/image_comparison.py which is a [marimo](https://marimo.io/) notebook (basically jupyter notebook)
That file contains a whole bunch of experiments of ways to encode this information and little UI elements to play with

As we hone in on better algorithms and the versions we will use, other files may appear for building optimized versions and scripts for integrating with the rest of payload software
