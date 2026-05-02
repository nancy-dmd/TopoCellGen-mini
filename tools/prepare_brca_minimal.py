import argparse
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--canvas-size", type=int, default=256)
    return parser.parse_args()


def stamp_square(canvas, y, x, channel):
    y0 = max(0, y - 1)
    y1 = min(canvas.shape[0], y + 2)
    x0 = max(0, x - 1)
    x1 = min(canvas.shape[1], x + 2)
    canvas[y0:y1, x0:x1, channel] = 1


def convert_label_file(label_path, output_path, canvas_size):
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            y, x, cls_id = map(int, line.split())
            if 0 <= y < canvas_size and 0 <= x < canvas_size and 1 <= cls_id <= 3:
                stamp_square(canvas, y, x, cls_id - 1)
    np.save(output_path, canvas)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.split_file, "r") as f:
        names = [line.strip() for line in f if line.strip()]

    if args.limit > 0:
        names = names[: args.limit]

    for image_name in names:
        stem = os.path.splitext(image_name)[0]
        label_name = f"{stem}_gt_class_coords.txt"
        label_path = os.path.join(args.labels_dir, label_name)
        output_path = os.path.join(args.output_dir, f"{stem}.npy")
        convert_label_file(label_path, output_path, args.canvas_size)
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()
