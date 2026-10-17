# split_dataset.py — run from the folder that contains dataset/
import shutil
from pathlib import Path

ds = Path("dataset")
CLASSES = ["jeep", "tesla"]
VAL_PER_CLASS = 4

for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    (ds / sub).mkdir(parents=True, exist_ok=True)

for c in CLASSES:
    imgs = sorted((ds / "images").glob(f"*{c}_*.jpg"))   # matches names with/without hash prefix
    for i, img in enumerate(imgs):
        split = "train" if i < len(imgs) - VAL_PER_CLASS else "val"
        lbl = ds / "labels" / f"{img.stem}.txt"
        shutil.move(str(img), str(ds / "images" / split / img.name))
        if lbl.exists():
            shutil.move(str(lbl), str(ds / "labels" / split / lbl.name))

for split in ["train", "val"]:
    ni = len(list((ds / "images" / split).glob("*.jpg")))
    nl = len(list((ds / "labels" / split).glob("*.txt")))
    print(f"{split}: {ni} images, {nl} labels")
