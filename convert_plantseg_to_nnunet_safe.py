# convert_plantseg_to_nnunet_safe.py
import os
import sys
import argparse
from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--plantseg_root", required=True, help="Path to Plantseg root (contains images/ and annotations/)")
parser.add_argument("--out_root", required=True, help="Path to nnUNet project root (will create nnUNet_raw/Dataset501_PlantSeg)")
parser.add_argument("--max_size", type=int, default=1024, help="Max image side length (keeps aspect ratio)")
parser.add_argument("--mode", choices=["train","all"], default="all", help="Convert 'train' only or 'all' (train+val+test)")
args = parser.parse_args()

PLANT = Path(args.plantseg_root)
OUTROOT = Path(args.out_root)
OUT = OUTROOT / "nnUNet_raw" / "Dataset501_PlantSeg"
IMG_OUT = OUT / "imagesTr"
MASK_OUT = OUT / "labelsTr"
IMG_TS_OUT = OUT / "imagesTs"
IMG_OUT.mkdir(parents=True, exist_ok=True)
MASK_OUT.mkdir(parents=True, exist_ok=True)
IMG_TS_OUT.mkdir(parents=True, exist_ok=True)

def find_folder(p: Path, candidates):
    for c in candidates:
        q = p / c
        if q.exists() and q.is_dir():
            return q
    return None

# try common locations
images_dir = find_folder(PLANT, ["images", "Images", "imgs", "imgs_train"])
ann_dir = find_folder(PLANT, ["annotations", "Annotations", "masks"])
if images_dir is None:
    print("ERROR: couldn't find an images/ folder under", PLANT)
    sys.exit(1)
if ann_dir is None:
    print("ERROR: couldn't find an annotations/ folder under", PLANT)
    sys.exit(1)

# inside images_dir there are likely train/val/test subfolders
def get_subdir(base, name):
    for n in [name, name.lower(), name.upper()]:
        p = base / n
        if p.exists(): return p
    return None

train_img_dir = get_subdir(images_dir, "train")
val_img_dir   = get_subdir(images_dir, "val")
test_img_dir  = get_subdir(images_dir, "test")

train_ann_dir = get_subdir(ann_dir, "train")
val_ann_dir   = get_subdir(ann_dir, "val")
test_ann_dir  = get_subdir(ann_dir, "test")

# helper: resize preserving aspect ratio
def resize_keep_ar(img, max_side, interp):
    h,w = img.shape[:2]
    scale = min(max_side/h, max_side/w, 1.0)
    new_w, new_h = int(w*scale), int(h*scale)
    return cv2.resize(img, (new_w, new_h), interpolation=interp)

counter = 1
def convert_pair(img_path, mask_path, out_img_path, out_mask_path, max_size):
    img = cv2.imread(str(img_path))
    if img is None:
        print("WARN: cannot read image", img_path)
        return False
    # mask may be json or png; we expect png mask per PlantSeg
    if not mask_path.exists():
        print("WARN: mask not found for", img_path.name)
        return False
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("WARN: cannot read mask", mask_path)
        return False
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8) * 1  # ensure 0/1
    img2 = resize_keep_ar(img, max_size, cv2.INTER_AREA)
    mask2 = resize_keep_ar(mask, max_size, cv2.INTER_NEAREST)
    cv2.imwrite(str(out_img_path), img2)
    cv2.imwrite(str(out_mask_path), mask2)
    return True

# process function for a subfolder
def process_folder(img_folder, ann_folder, is_train=True):
    global counter
    if img_folder is None:
        return 0
    files = sorted([p for p in img_folder.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png")])
    converted = 0
    for p in tqdm(files, desc=f"Processing {img_folder.name}"):
        # find mask: try same basename + .png in ann_folder
        base = p.stem
        # common PlantSeg uses same basename (without _0000)
        cand_masks = [
            ann_folder / (base + ".png"),
            ann_folder / (base + ".PNG"),
            ann_folder / (base + ".jpg"),
            ann_folder / (base + ".json")  # fallback, but we don't parse json polygons here
        ]
        maskp = None
        for c in cand_masks:
            if c.exists():
                maskp = c
                break
        if maskp is None:
            # sometimes annotation files are named differently - try any file with same index
            # skip if not found
            print("Missing mask for", p.name, "- skipping")
            continue
        if is_train:
            cid = f"plant_{counter:04d}"
            out_img = IMG_OUT / f"{cid}_0000{p.suffix.lower()}"
            out_mask = MASK_OUT / f"{cid}.png"
            ok = convert_pair(p, maskp, out_img, out_mask, args.max_size)
            if ok:
                counter += 1
                converted += 1
        else:
            # test/val → copy into imagesTs with unique name
            cid = f"plant_ts_{counter:04d}"
            out_img = IMG_TS_OUT / f"{cid}_0000{p.suffix.lower()}"
            # we do not create masks for imagesTs
            img = cv2.imread(str(p))
            img2 = resize_keep_ar(img, args.max_size, cv2.INTER_AREA)
            cv2.imwrite(str(out_img), img2)
            counter += 1
            converted += 1
    return converted

total = 0
# train
total += process_folder(train_img_dir, train_ann_dir, is_train=True)
if args.mode == "all":
    # include val images (as training if you prefer), or put val into imagesTr as well
    if val_img_dir and val_ann_dir:
        print("Also converting val -> training (you may instead use val for separate validation).")
        total += process_folder(val_img_dir, val_ann_dir, is_train=True)
    # test into imagesTs
    if test_img_dir:
        print("Converting test images into imagesTs (no masks).")
        total += process_folder(test_img_dir, test_ann_dir, is_train=False)

print(f"Converted {total} images. Output in: {OUT}")
# write dataset.json with correct channel names and numTraining
ds = {
    "channel_names": {"0":"R","1":"G","2":"B"},
    "labels": {"background":0,"disease":1},
    "numTraining": total,
    "file_ending": ".jpg",
    "dataset_name": "PlantSeg",
    "description": "Converted PlantSeg -> nnUNet_raw Dataset501_PlantSeg"
}
with open(OUT / "dataset.json","w", encoding="utf-8") as f:
    json.dump(ds, f, indent=2)
print("Wrote dataset.json with numTraining =", total)
