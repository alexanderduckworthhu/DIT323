from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO
from PIL import Image
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "dataset" / "images"
DEMO_FILES = {
    "test1.jpg": ROOT / "test1.jpg",
    "test2.jpg": ROOT / "test2.jpg",
}

app = FastAPI(title="Corgi vs Chihuahua")

# allow browser frontends on other origins to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO(str(ROOT / "best.pt"))


def detections(img: Image.Image):
    result = model(img)[0]
    return [
        {"class": result.names[int(b.cls)], "conf": float(b.conf), "box": b.xyxy[0].tolist()}
        for b in result.boxes
    ]


def sample_path(split: str, name: str) -> Path:
    if "/" in name or ".." in name or not name.endswith(".jpg"):
        raise HTTPException(status_code=404, detail="image not found")
    if split == "demo":
        path = DEMO_FILES.get(name)
    elif split in {"train", "val"}:
        path = IMAGE_ROOT / split / name
    else:
        path = None
    if path is None:
        raise HTTPException(status_code=404, detail="image not found")
    path = path.resolve()
    if split == "demo":
        if path.parent != ROOT:
            raise HTTPException(status_code=404, detail="image not found")
    elif IMAGE_ROOT.resolve() not in path.parents:
        raise HTTPException(status_code=404, detail="image not found")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return path


@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")


@app.get("/samples")
def samples():
    items = []
    for split in ("train", "val"):
        folder = IMAGE_ROOT / split
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.jpg")):
            items.append({
                "name": path.name,
                "split": split,
                "url": f"/samples/{split}/{path.name}",
            })
    for name in ("test1.jpg", "test2.jpg"):
        if (ROOT / name).is_file():
            items.append({"name": name, "split": "demo", "url": f"/samples/demo/{name}"})
    return items


@app.get("/samples/{split}/{name}")
def sample_file(split: str, name: str):
    return FileResponse(sample_path(split, name))


@app.post("/samples/{split}/{name}/predict")
def predict_sample(split: str, name: str):
    img = Image.open(sample_path(split, name)).convert("RGB")
    return detections(img)


@app.post("/predict")
async def predict(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    result = model(img)[0]
    return [
        {"class": result.names[int(b.cls)], "conf": float(b.conf), "box": b.xyxy[0].tolist()}
        for b in result.boxes
    ]
