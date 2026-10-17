# DIT323 model apps

Each YOLO app is one local server. That server is the API and the web page. Open only one of these at a time on port 8000, or start the second app on port 8001.

Python 3.13 is required. Each project folder already has a virtual environment on this machine. A fresh clone does not include it. Create one inside the project folder before the start command:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install ultralytics fastapi uvicorn python-multipart
```

For the customer buy predictor, install this set instead:

```bash
pip install pandas numpy scikit-learn joblib fastapi "uvicorn[standard]"
```

## Corgi vs Chihuahua

From the `DIT323` folder:

```bash
cd yolo-project
source .venv/bin/activate
uvicorn main:app --reload
```

- App: http://127.0.0.1:8000/
- API docs: http://127.0.0.1:8000/docs

Click a photo in the gallery. The page sends that dataset image to the API and draws the box. You can also choose a file from your computer.

`POST /predict` accepts an uploaded image. `POST /samples/{split}/{name}/predict` runs a photo that is already in the dataset (`train`, `val`, or `demo`).

## Jeep vs Tesla

Stop the corgi server first, or use another port. From the `DIT323` folder:

```bash
cd yolo-project-cars
source .venv/bin/activate
uvicorn main:app --reload --port 8001
```

- App: http://127.0.0.1:8001/
- API docs: http://127.0.0.1:8001/docs

Use the gallery the same way. Full-car photos are the ones to demo. The last Jeep thumbnail is a close crop of a door and can draw extra boxes.

## Customer buy predictor

This one keeps the API and the page in two terminals.

```bash
cd model-serving
source .venv/bin/activate
uvicorn app:app --reload
```

In a second terminal:

```bash
cd web
python3 -m http.server 5500
```

- App: http://127.0.0.1:5500/
- API docs: http://127.0.0.1:8000/docs

Do not run this API at the same time as a YOLO app on port 8000.
