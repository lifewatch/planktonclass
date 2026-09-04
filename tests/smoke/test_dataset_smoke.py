from pathlib import Path

import cv2
import numpy as np

from planktonclass.data_utils import (
    create_data_splits,
    load_class_names,
    load_data_splits,
    split_file_has_entries,
)


def test_dataset_split_generation_and_loading_smoke(tmp_path):
    images_dir = tmp_path / "images"
    splits_dir = tmp_path / "dataset_files"

    for class_name, filenames in {
        "ClassA": ["a1.jpg", "a2.png"],
        "ClassB": ["b1.image"],
    }.items():
        class_dir = images_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            image_path = class_dir / filename
            pixels = np.zeros((4, 4, 3), dtype=np.uint8)
            if image_path.suffix == ".image":
                success, encoded = cv2.imencode(".jpg", pixels)
                assert success
                image_path.write_bytes(encoded.tobytes())
            else:
                assert cv2.imwrite(str(image_path), pixels)

        (class_dir / "metadata.csv").write_text("sample,value\n1,2\n")
        (class_dir / "notes.txt").write_text("not an image")

    create_data_splits(str(splits_dir), str(images_dir), split_ratios=[1, 0, 0])

    assert split_file_has_entries(str(splits_dir), "train")
    assert load_class_names(str(splits_dir)).tolist() == ["ClassA", "ClassB"]

    X_train, y_train = load_data_splits(str(splits_dir), str(images_dir), "train")
    relative_paths = sorted(
        str(Path(path).relative_to(images_dir)).replace("\\", "/") for path in X_train
    )
    assert relative_paths == ["ClassA/a1.jpg", "ClassA/a2.png", "ClassB/b1.image"]
    assert y_train.tolist() == [0, 0, 1]
