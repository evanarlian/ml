from argparse import ArgumentParser
from io import BytesIO
from itertools import accumulate
from multiprocessing import Process
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


def hehe(
    dataset: Dataset,
    start_idx: int,
    end_idx: int,
    min_size: int,
    split_dir: Path,
    proc_id: int,
):
    for i in tqdm(
        range(start_idx, end_idx), position=proc_id, desc=f"{split_dir.name}:{proc_id}"
    ):
        batch = dataset[i]
        bio = BytesIO(batch["image"]["bytes"])
        im = Image.open(bio).convert("RGB")
        # resize smaller side to min_size
        w, h = im.size
        if w < h:
            new_w = min_size
            new_h = int(min_size * h / w)
        else:
            new_h = min_size
            new_w = int(min_size * w / h)
        im = im.resize((new_w, new_h))
        # real filename might not be .JPEG, better force them
        filename = batch["image"]["path"].split(".")[0] + ".JPEG"
        label = "_1" if batch["label"] == -1 else f"{batch['label']:03d}"
        im.save(split_dir / label / filename)


def get_ranges(n: int, n_splits: int):
    part = n // n_splits
    temp = [part + (i < n % n_splits) for i in range(n_splits)]
    cumsum = [0] + list(accumulate(temp))
    return list(zip(cumsum[:-1], cumsum[1:]))


def download_imagenet(root_dir: Path, min_size: int, n_proc: int):
    # do not decode because we need the real filename
    ds = load_dataset("imagenet-1k").cast_column("image", HFImage(decode=False))

    # save train
    train_processes = []
    for i, (start_idx, end_idx) in enumerate(get_ranges(len(ds["train"]), n_proc)):
        p = Process(
            target=hehe,
            args=(ds["train"], start_idx, end_idx, min_size, root_dir / "train", i),
        )
        train_processes.append(p)
        p.start()
    for p in train_processes:
        p.join()

    # save val
    val_processes = []
    for i, (start_idx, end_idx) in enumerate(get_ranges(len(ds["val"]), n_proc)):
        p = Process(
            target=hehe,
            args=(ds["val"], start_idx, end_idx, min_size, root_dir / "val", i),
        )
        val_processes.append(p)
        p.start()
    for p in val_processes:
        p.join()

    # save test
    test_processes = []
    for i, (start_idx, end_idx) in enumerate(get_ranges(len(ds["test"]), n_proc)):
        p = Process(
            target=hehe,
            args=(ds["test"], start_idx, end_idx, min_size, root_dir / "test", i),
        )
        test_processes.append(p)
        p.start()
    for p in test_processes:
        p.join()


def main(args):
    root_dir = Path("data") / f"imagenet_1k_resized_{args.min_size}"
    print(f"Saving dataset to {root_dir}, using {args.n_proc} processes")
    make_directories(root_dir)
    download_imagenet(root_dir, min_size=args.min_size, n_proc=args.n_proc)


if __name__ == "__main__":
    # NOTE you must accept to imagenet's permission to be able to access
    parser = ArgumentParser()
    parser.add_argument("--min_size", type=int, default=256, help="Resize min size to")
    parser.add_argument("--n_proc", type=int, default=4, help="Num proc for processing")
    args = parser.parse_args()
    main(args)
