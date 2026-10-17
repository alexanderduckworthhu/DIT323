# check_export.py — run from the folder that contains dataset/
from pathlib import Path

ds = Path("dataset")
# 1) class order: class id = line number (starts at 0)
classes = [c.strip() for c in (ds / "classes.txt").read_text().splitlines() if c.strip()]
print("class order:", {i: c for i, c in enumerate(classes)})

# 2) image <-> label stems must match
imgs = {p.stem for p in (ds / "images").glob("*.jpg")}
lbls = {p.stem for p in (ds / "labels").glob("*.txt")}
if imgs == lbls:
    print(f"OK: {len(imgs)} images all have matching labels")
else:
    print("image without label:", sorted(imgs - lbls)[:5])
    print("label without image:", sorted(lbls - imgs)[:5])
