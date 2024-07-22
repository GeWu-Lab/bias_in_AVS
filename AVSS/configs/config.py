# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
 
import os

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2  # type: ignore

import argparse
import json
import os
from typing import Any, Dict, List

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    default='./input',
    # required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--data_ver",
    type=str,
    default='',
    # required=True,
    help="data_ver v1m, v2.",
)

parser.add_argument(
    "--output",
    type=str,
    default='./output',
    # required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    default='vit_b',
    # required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    # default='./checkpoint/sam_vit_b_01ec64.pth',
    default='./checkpoint/sam_vit_b_01ec64.pth',
    # required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)
parser.add_argument("--lgct_l", type=str, default='add', help='seq local proj. [add, w_add, tf, attn]')
parser.add_argument("--lgct_g", type=bool, default=True, help='seq global proj.')
parser.add_argument("--lr", type=float, default=1e-3, help='lr to fine tuning adapters.')
parser.add_argument("--epochs", type=int, default=240, help='epochs to fine tuning adapters.')
parser.add_argument("--loss", type=str, default='bce', help='bce | mse | focal | focal & dice.')
parser.add_argument("--multi", default=False, action='store_true', help='whether generate 3 masks for each frame.')
parser.add_argument("--train", default=False, action='store_true', help='start train?')
parser.add_argument("--val", type=str, default=None, help='type: str; val | test')  # NOTE: for test and val.
parser.add_argument("--test", default=False, action='store_true', help='start test?')
parser.add_argument("--show", default=False, action='store_true', help='show mask?')
parser.add_argument("--device", type=str, default="cuda:0", help="The device to run generation on.")
parser.add_argument("--gpu_id", type=str, default="0", help="The GPU device to run generation on.")
parser.add_argument("--x", type=int, default="-1", help="the epoch to select for testing.")
parser.add_argument("--tune_v", type=int, default=12, help="the #id of block for image encoder adapter tuning stating.")
parser.add_argument("--save_mask", default=False, action='store_true', help="whether save the test set mask.")
parser.add_argument("--db", default=None, help="audio db exp.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

parser.add_argument("--run", type=str, default='train', help="train, test")

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
