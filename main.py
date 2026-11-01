import argparse
from pathlib import Path

import torch

from detect import Detect
from net import FCNet, Net

NETS = {
    "FCNet": FCNet,
    "Net": Net,
}


def main():
    args = parser().parse_args()

    net = load_net(args.model, args.weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    detector = Detect(
        net=net,
        input_max_side=args.size,
        step_size=args.step,
        scaling_sizes=args.scales,
        certainty_filter=args.certainty,
        knn=args.knn,
    )
    exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

    input = Path(args.input)
    paths: list[Path]
    if input.is_dir():
        paths = [
            p
            for p in input.iterdir()
            if not (p.is_dir() or p.suffix.lower() not in exts)
        ]
    else:
        paths = [input]

    detected = []
    for path in paths:
        print(f"detecting faces in {path}...")
        detected.append(detector.detect_faces(path))

    out = Path(args.out)
    if out.is_dir():
        for p, d in zip(paths, detected):
            d.save(out / p.name)
    else:
        detected[0].save(out)
    print("Done!")


def load_net(name: str, weights: str):
    if name not in NETS:
        raise ValueError(f"{name} is not a supported model: try one of {NETS.keys}")
    net = NETS[name]()
    state = torch.load(weights, map_location=torch.device("cpu"))
    net.load_state_dict(state)
    net.eval()
    return net


def parser():

    parser = argparse.ArgumentParser(
        prog="face detector", description="detect faces in images"
    )

    parser.add_argument("input")

    parser.add_argument("-o", "--out", required=True)

    parser.add_argument("-m", "--model", choices=NETS, default="FCNet")
    parser.add_argument("-w", "--weights", required=True)
    parser.add_argument(
        "-c", "--certainty", type=float, default=Detect.DEFAULT_CERTAINTY
    )
    parser.add_argument("--knn", type=int, default=Detect.DEFAULT_KNN)
    parser.add_argument("-s", "--step", type=int, default=Detect.DEFAULT_STEP)
    parser.add_argument("--size", type=int, default=Detect.DEFAULT_SIZE)
    parser.add_argument(
        "--scales", type=float, nargs="+", default=Detect.DEFAULT_SCALING_SIZES
    )
    return parser


if __name__ == "__main__":
    main()
