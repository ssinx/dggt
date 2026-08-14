"""Persistent stdin/stdout SegFormer worker for in-memory sky masks."""

import argparse
import contextlib
import pickle
import struct
import sys

import cv2
import numpy as np


def read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"Pipe closed with {remaining} bytes still expected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_message(stream):
    first_byte = stream.read(1)
    if not first_byte:
        return None
    header = first_byte + read_exact(stream, 7)
    size = struct.unpack("!Q", header)[0]
    payload = read_exact(stream, size)
    return pickle.loads(payload)


def write_message(stream, value):
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("!Q", len(payload)))
    stream.write(payload)
    stream.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segformer_path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.segformer_path)
    # Keep stdout exclusively for the binary protocol.
    with contextlib.redirect_stdout(sys.stderr):
        from mmseg.apis import inference_segmentor, init_segmentor
        model = init_segmentor(args.config, args.checkpoint, device=args.device)

    while True:
        request = read_message(sys.stdin.buffer)
        if request is None or request == "stop":
            break
        try:
            masks = []
            with contextlib.redirect_stdout(sys.stderr):
                for encoded_image in request:
                    image = cv2.imdecode(np.frombuffer(encoded_image, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("Failed to decode a Waymo camera image")
                    segmentation = np.asarray(inference_segmentor(model, image)[0])
                    masks.append((segmentation == 10).astype(np.uint8))
            write_message(sys.stdout.buffer, {"masks": masks})
        except Exception as error:
            write_message(sys.stdout.buffer, {"error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
