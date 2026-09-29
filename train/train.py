"""Configuration-driven LeRobot/EMIX training entrypoint."""

import sys

from embodied_mixup.cli import main


if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
