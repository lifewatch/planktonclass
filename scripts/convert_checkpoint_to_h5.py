"""
Convert a Keras checkpoint to H5 for older Windows TensorFlow runtimes.
"""

import argparse
from pathlib import Path

from tensorflow.keras.models import load_model

from planktonclass import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Path to a .keras or .h5 checkpoint")
    parser.add_argument(
        "--output",
        help="Optional output .h5 path. Defaults to the source path with .h5 suffix.",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Checkpoint not found: {source}")

    if args.output:
        output = Path(args.output).resolve()
    else:
        output = source.with_suffix(".h5")

    output.parent.mkdir(parents=True, exist_ok=True)
    model = load_model(source, custom_objects=utils.get_custom_objects(), compile=False)
    model.save(output, include_optimizer=False)
    print(f"Converted checkpoint written to: {output}")


if __name__ == "__main__":
    main()
