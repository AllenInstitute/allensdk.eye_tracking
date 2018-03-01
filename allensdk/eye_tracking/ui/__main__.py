import sys
import argparse
import json
import logging
from allensdk.eye_tracking.ui.qt import QtWidgets, ViewerWindow
from allensdk.eye_tracking import _schemas


def load_config(config_file):
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
    except Exception as e:
        logging.error(e)
        config = None
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--config_file", type=str, default="")
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left
    config = None
    if args.config_file:
        config = load_config(args.config_file)
    app = QtWidgets.QApplication([])
    w = ViewerWindow(_schemas.InputParameters, args.profile, config)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()