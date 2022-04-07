#!/usr/bin/env python3

import argparse
import os
import tempfile
from typing import Iterator, List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from rich.progress import track
from rich import print


def readrgb(fname: str, offset: int) -> Iterator[Tuple[int, int, int]]:
    with open(fname, "rb") as file:
        file.seek(offset)
        while True:
            result = file.read(3)
            if len(result) < 3:
                break
            r = result[0]
            g = result[1]
            b = result[2]
            yield (r, g, b)


def readhex(fname: str, offset: int) -> Iterator[bytes]:
    with open(fname) as file:
        file.seek(2 + (offset * 2))  # Skip initial '3.' Each digit is 2 bytes.
        while True:
            result = str(file.read(2))
            if len(result) < 2:
                break
            yield int(result[0:2], 16).to_bytes(1, byteorder="big", signed=False)


def readimage(fname: str, offset: int, width: int, height: int) -> Image:
    iter = readrgb(fname, offset)
    img = Image.new(mode="RGB", size=(width, height))
    for y in range(height):
        for x in range(width):
            r, g, b = next(iter)
            img.putpixel((x, y), (r, g, b))
            # print(f"Pixel {x},{y} -> {r:02x} {g:02x} {b:02x}")
    return img


def postprocess(img: Image) -> np.ndarray:
    # Post-process image.
    arr = np.asarray(img, dtype="float32")
    # Converts RGB pixel value from [0 - 255] to float array [-1.0 - 1.0]
    arr -= [127.0, 127.0, 127.0]
    arr /= [128.0, 128.0, 128.0]
    return arr


def classify(
    ort_sess: ort.InferenceSession, img: Image, labels: List[str]
) -> List[Tuple[str, float]]:
    arr = postprocess(img)
    results = ort_sess.run(["Softmax:0"], {"images:0": [arr]})[0]
    result = reversed(results[0].argsort()[-5:])
    return [(labels[r], results[0][r]) for r in result]


def convert_hex_to_binary(infile: str, outfile: str):
    file_size = os.path.getsize(infile)
    # First two characters in the file are '3.'
    # and each hex digit is 2 bytes in ASCII.
    num_output_bytes = (file_size - 2) // 2
    iter = readhex(infile, 0)
    with open(outfile, "wb") as output:
        for b in track(
            iter, total=num_output_bytes, description="Converting to binary..."
        ):
            output.write(b)


def find_images(args: argparse.Namespace):
    # Load model.
    print(f"Using model {args.model}")
    ort_sess = ort.InferenceSession(args.model)

    # Read classes.
    with open(args.labels) as labels_file:
        labels = labels_file.readlines()
        print(f"Read {len(labels)} labels from {args.labels}")

    with tempfile.NamedTemporaryFile("wb") as outfile:
        # Convert input file to binary.
        convert_hex_to_binary(args.INFILE, outfile.name)
        file_size = os.path.getsize(outfile.name)
        for offset in track(range(file_size), "Reading images..."):
            img = readimage(outfile.name, offset, args.width, args.height)
            result = classify(ort_sess, img, labels)
            for (label, prob) in result:
                if prob >= args.threshold:
                    print(f"Offset {offset} detected: {prob} {label}")


def show_image(args):
    print(f"Using model {args.model}")
    ort_sess = ort.InferenceSession(args.model)

    # Read classes.
    with open(args.labels) as labels_file:
        labels = labels_file.readlines()
        print(f"Read {len(labels)} labels from {args.labels}")

    with tempfile.NamedTemporaryFile("wb") as outfile:
        convert_hex_to_binary(args.INFILE, outfile.name)
        img = readimage(outfile.name, args.show, args.width, args.height)
        img.show()
        result = classify(ort_sess, img, labels)
        for (label, prob) in result:
            print(f"Offset {args.show} detected: {prob} {label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("INFILE")
    parser.add_argument("--width", default=224, type=int)
    parser.add_argument("--height", default=224, type=int)
    parser.add_argument("--model", default="efficientnet-lite4-11-int8.onnx")
    parser.add_argument("--labels", default="imagenet_classes.txt")
    parser.add_argument("--threshold", default=0.1, type=float)
    parser.add_argument("--show", default=None, type=int)
    args = parser.parse_args()

    if args.show is not None:
        show_image(args)
    else:
        find_images(args)


if __name__ == "__main__":
    main()
