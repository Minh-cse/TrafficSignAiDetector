from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import base64
import logging
from typing import Optional, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"

# Load model with error handling
model: Optional[YOLO] = None
try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")

# Classes map
CLASSES_MAP: Dict[int, str] = {
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)',
    2:'Speed limit (50km/h)', 3:'Speed limit (60km/h)',
    4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)',
    8:'Speed limit (120km/h)', 9:'No passing',
    10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
    12:'Priority road', 13:'Yield', 14:'Stop',
    15:'No vehicles', 16:'Veh > 3.5 tons prohibited',
    17:'No entry', 18:'General caution',
    19:'Dangerous curve left', 20:'Dangerous curve right',
    21:'Double curve', 22:'Bumpy road',
    23:'Slippery road', 24:'Road narrows on the right',
    25:'Road work', 26:'Traffic signals',
    27:'Pedestrians', 28:'Children crossing',
    29:'Bicycles crossing', 30:'Beware of ice/snow',
    31:'Wild animals crossing', 32:'End speed + passing limits',
    33:'Turn right ahead', 34:'Turn left ahead',
    35:'Ahead only', 36:'Go straight or right',
    37:'Go straight or left', 38:'Keep right',
    39:'Keep left', 40:'Roundabout mandatory',
    41:'End of no passing', 42:'End no passing veh > 3.5 tons'
}

# Configuration constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 10
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
FONT_SCALE = 0.5
TEXT_THICKNESS = 1

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}

def process_detections(results, original_img: np.ndarray):
    detections = []
    img_with_boxes = original_img.copy()

    boxes = getattr(results, "boxes", None)
    if boxes is None:
        # fallback to plot if boxes missing
        plotted = results.plot() if hasattr(results, "plot") else None
        if plotted is not None:
            img_with_boxes = cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR) if plotted.ndim == 3 and plotted.shape[2] == 3 else plotted
        success, buffer = cv2.imencode(".jpg", img_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image")
        return [], base64.b64encode(buffer.tobytes()).decode("utf-8")

    for box in boxes:
        # get coordinates robustly
        coords = None
        try:
            # box.xyxy may be tensor-like
            xy = box.xyxy[0] if hasattr(box.xyxy, "__len__") and len(box.xyxy) > 0 else box.xyxy
            coords = [float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])]
        except Exception:
            try:
                coords = [float(v) for v in getattr(box.xyxy, "tolist", lambda: box.xyxy)()]
            except Exception:
                try:
                    coords = list(map(float, box.xyxy.cpu().numpy().flatten()))
                except Exception:
                    continue

        x1, y1, x2, y2 = map(int, coords)
        h_img, w_img = img_with_boxes.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)

        # safe class id and confidence
        try:
            cls_val = box.cls if not hasattr(box.cls, "__len__") else box.cls[0]
            cls_id = int(cls_val.item()) if hasattr(cls_val, "item") else int(cls_val)
        except Exception:
            cls_id = -1
        try:
            conf_val = box.conf if not hasattr(box.conf, "__len__") else box.conf[0]
            conf = float(conf_val.item()) if hasattr(conf_val, "item") else float(conf_val)
        except Exception:
            conf = 0.0

        class_name = CLASSES_MAP.get(cls_id, "Unknown")

        # draw box
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        # label text (use name + confidence)
        label = f"{class_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)
        # background rectangle for text
        yy = max(0, y1 - th - 6)
        cv2.rectangle(img_with_boxes, (x1, yy), (x1 + tw + 6, y1), BOX_COLOR, -1)
        cv2.putText(img_with_boxes, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), TEXT_THICKNESS, cv2.LINE_AA)

        detections.append({
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": round(conf, 3)
        })
        logger.info(f"Detected sign: {class_name} (id={cls_id}, conf={conf:.3f})")

    # encode to base64
    success, buffer = cv2.imencode(".jpg", img_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    encoded_image = base64.b64encode(buffer.tobytes()).decode("utf-8")

    return detections, encoded_image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict traffic signs from uploaded image"""
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    contents = await file.read()

    # 🔒 SAFETY CHECK 1: empty upload
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file received")

    # 🔒 SAFETY CHECK 2: file size limit
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")

    # Decode image from bytes
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # 🔒 SAFETY CHECK 3: invalid image
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Optimize image size if too large (work on a copy)
    height, width = img.shape[:2]
    if width > 1280:
        scale = 1280 / width
        new_width = 1280
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    try:
        results = model.predict(
            source=img,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DETECTIONS,
            agnostic_nms=True,
            verbose=False,
            device='cpu'  # Explicitly set device
        )[0]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    try:
        detections, encoded = process_detections(results, img)
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail="Post-processing failed")

    logger.info(f"Detected {len(detections)} signs")

    return {
        "detections": sorted(detections, key=lambda x: x["confidence"], reverse=True),
        "processed_image": encoded
    }
