import time
from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset


def main(root_dir: Path, username: str):
    # load real imagenet for reference
    imagenet_labels = load_dataset(
        "imagenet-1k", split="validation", streaming=True
    ).features["label"]

    # need to use data_files since data_dir will cause duplicate filenames
    # (only on special case, in this case: duplicate in validation only)
    t0 = time.perf_counter()
    ds = load_dataset(
        "imagefolder",
        data_files={
            "train": f"{root_dir}/train/**",
            "val": f"{root_dir}/val/**",
            "test": f"{root_dir}/test/**",
        },
    )
    dt = time.perf_counter() - t0
    print(f"Reading dataset took {dt:.3f}s")

    # cast dummy class labels to real imagenet label
    ds["train"] = ds["train"].cast_column("label", imagenet_labels)
    ds["val"] = ds["val"].cast_column("label", imagenet_labels)
    ds["test"] = ds["test"].cast_column("label", imagenet_labels)

    # force set test label to -1
    ds["test"] = ds["test"].map(lambda row: row | {"label": -1})

    ds.push_to_hub(f"{username}/{root_dir.name}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--folder",
        type=Path,
        default="data/imagenet_1k_resized_256",
        help="Imagenet folder to upload to HF hub.",
    )
    parser.add_argument(
        "--username", type=str, default="evanarlian", help="HF username."
    )
    args = parser.parse_args()
    if not args.folder.exists():
        raise ValueError(f"Folder does not exist: {args.folder}")
    print(f"Dataset will be pushed to {args.username}/{args.folder.name}")
    main(args.folder, args.username)
