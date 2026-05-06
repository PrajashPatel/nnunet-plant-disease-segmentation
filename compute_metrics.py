import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------
# PATHS
# ---------------------------
GT_DIR = r"nnUNet_raw/Dataset501_PlantSeg/labelsTr"
PRED_DIR = r"nnUNet_results/Dataset501_PlantSeg/nnUNetTrainer__nnUNetResEncUNetMPlans__2d/fold_0/validation"

print("Ground-truth DIR:", GT_DIR)
print("Prediction DIR:", PRED_DIR)

pred_files = sorted([f for f in os.listdir(PRED_DIR) if f.endswith(".npz")])

# ---------------------------
# Metric functions
# ---------------------------
def dice_score(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return (2 * intersection + eps) / (pred.sum() + gt.sum() + eps)

def iou_score(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (intersection + eps) / (union + eps)

def precision_score(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    return (tp + eps) / (tp + fp + eps)

def recall_score(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return (tp + eps) / (tp + fn + eps)

def pixel_accuracy(pred, gt):
    return (pred == gt).sum() / pred.size

# ---------------------------
# Metric accumulators
# ---------------------------
all_dice = []
all_iou = []
all_precision = []
all_recall = []
all_accuracy = []

# ---------------------------
# Evaluation Loop
# ---------------------------
for f in tqdm(pred_files, desc="Evaluating", ncols=100):
    pred_path = os.path.join(PRED_DIR, f)
    case_id = f.replace(".npz", "")
    gt_path = os.path.join(GT_DIR, case_id + ".png")

    if not os.path.exists(gt_path):
        print("Missing GT for:", case_id)
        continue

    # Load prediction from nnU-Net
    pred_npz = np.load(pred_path)["probabilities"]   # shape: [C, H, W]
    pred_seg = np.argmax(pred_npz, axis=0).astype(np.uint8)

    # Load GT
    gt = np.array(Image.open(gt_path)).astype(np.uint8)

    classes = np.unique(gt)

    dice_list = []
    iou_list = []
    precision_list = []
    recall_list = []

    # Per-class metrics
    for cls in classes:
        pred_bin = (pred_seg == cls).astype(np.uint8)
        gt_bin   = (gt == cls).astype(np.uint8)

        dice_list.append(dice_score(pred_bin, gt_bin))
        iou_list.append(iou_score(pred_bin, gt_bin))
        precision_list.append(precision_score(pred_bin, gt_bin))
        recall_list.append(recall_score(pred_bin, gt_bin))

    # macro averages for this case
    all_dice.append(np.mean(dice_list))
    all_iou.append(np.mean(iou_list))
    all_precision.append(np.mean(precision_list))
    all_recall.append(np.mean(recall_list))

    # pixel accuracy (global, not per class)
    all_accuracy.append(pixel_accuracy(pred_seg, gt))

# ---------------------------
# Final summary
# ---------------------------
print("\n==========================")
print("FINAL NNUNET METRICS")
print("==========================")
print("Macro Dice:", np.mean(all_dice))
print("Macro IoU:", np.mean(all_iou))
print("Precision:", np.mean(all_precision))
print("Recall:", np.mean(all_recall))
print("Pixel Accuracy:", np.mean(all_accuracy))
print("==========================\n")
