from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path

from datasets import Dataset, load_dataset
from datasets import Image as HFImage
from PIL import Image
from tqdm.auto import tqdm


def make_directories(root_dir: Path):
    root_dir.mkdir(parents=True, exist_ok=True)
    train_dir = root_dir / "train"
    val_dir = root_dir / "val"
    test_dir = root_dir / "test"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    for i in range(1000):
        folder_name = f"{i:03d}"
        (train_dir / folder_name).mkdir(exist_ok=True)
        (val_dir / folder_name).mkdir(exist_ok=True)
    (test_dir / "_1").mkdir(exist_ok=True)  # placeholder for no label


def save_single_dataset(split_dir: Path, dataset: Dataset, min_size: int):
    for batch in tqdm(dataset):
        bio = BytesIO(batch["image"]["bytes"])
        im = Image.open(bio).convert("RGB")
        # thumbnail preserve aspect ratio and it is inplace
        im.thumbnail((min_size, min_size))
        # real filename might not be .JPEG, better force them
        filename = batch["image"]["path"].split(".")[0] + ".JPEG"
        label = "_1" if batch["label"] == -1 else f"{batch['label']:03d}"
        im.save(split_dir / label / filename)


def download_imagenet(root_dir: Path, min_size: int):
    # do not decode because we need the real filename
    ds = load_dataset("imagenet-1k", streaming=True).cast_column(
        "image", HFImage(decode=False)
    )
    train_ds = ds["train"]
    val_ds = ds["validation"]
    test_ds = ds["test"]

    train_dir = root_dir / "train"
    val_dir = root_dir / "val"
    test_dir = root_dir / "test"

    save_single_dataset(train_dir, train_ds, min_size)
    save_single_dataset(val_dir, val_ds, min_size)
    save_single_dataset(test_dir, test_ds, min_size)


def main(args):
    root_dir = Path("data/imagenet/")
    make_directories(root_dir)
    download_imagenet(root_dir, min_size=args.min_size)


if __name__ == "__main__":
    # NOTE you must accept to imagenet's permission to be able to access
    # the resize default is 256 because most common computer vision workflow is to
    # resize to 256, and then random crop to 224
    parser = ArgumentParser()
    parser.add_argument("--min_size", type=int, default=256, help="Resize min size to")
    args = parser.parse_args()
    main(args)
