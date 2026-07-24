"""Minimal CLI for running the saved OCRPipeline on one or more line images.

Usage:
    python app/infer.py --image data/VedOCR_kaggle_release/hf_format/test/page002_line001.png
    python app/infer.py --image path/to/img1.png path/to/img2.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.tokenizer import Tokenizer
from src.pipeline import OCRPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", nargs="+", required=True, help="path(s) to line image(s)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--pipeline", default=None, help="override path to the saved pipeline .pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = Tokenizer(cfg["data"]["vocab_path"])
    pipeline_path = args.pipeline or cfg["paths"]["pipeline"]

    pipeline = OCRPipeline.load(
        pipeline_path, tokenizer.n_classes,
        cfg["model"]["rnn_hidden"], cfg["model"]["rnn_layers"],
    )

    for path in args.image:
        pred = pipeline.predict(path)
        print(f"{path}\n  -> {pred if pred else '(empty prediction)'}\n")


if __name__ == "__main__":
    main()
