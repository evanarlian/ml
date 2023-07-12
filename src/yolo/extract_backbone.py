from argparse import ArgumentParser
from pathlib import Path

import pretrain_config as cfg
import torch
from yolo_model import YoloPretraining


def main(args):
    print(f"Load from {args.pretrained}")
    pretrained = YoloPretraining(cfg.N_CLASSES)
    pretrained.load_state_dict(torch.load(args.pretrained, map_location="cpu"))
    torch.save(pretrained.backbone.state_dict(), args.output)
    print(f"Backbone saved to {args.output}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "pretrained", type=Path, help="pretrained weights in accelerate_logs."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/yolo/backbone.pt"),
        help="filepath to output the backbone.",
    )
    args = parser.parse_args()
    main(args)
