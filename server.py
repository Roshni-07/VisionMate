"""
VisionMate – Flask API Server (Cloud + User Accounts Edition)
=============================================================
Interdisciplinary Project (1BPRJ208) | BMSIT&M | 2025-26

Each user has their own private face database.
Deploys to Render.com / Railway / any cloud.

Install: pip install flask flask-cors opencv-contrib-python easyocr torch torchvision numpy pillow gunicorn
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import sqlite3
import numpy as np
import json
import uuid
import threading
import os
from datetime import datetime
from pathlib import Path
from io import BytesIO
from PIL import Image
import base64
import socket

# ── DB path — use /data on cloud, local otherwise ────────────────────────────
DB_PATH = Path(os.environ.get("DB_PATH", "visionmate.db"))

# ── Lazy model loading — saves RAM on startup ─────────────────────────────────
YOLO    = None
YOLO_OK = False
OCR     = None
OCR_OK  = False

def get_yolo():
    global YOLO, YOLO_OK
    if YOLO is None:
        try:
            import torch
            YOLO    = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, trust_repo=True)
            YOLO.conf = 0.40
            YOLO.iou  = 0.45
            YOLO_OK   = True
            print("[Server] YOLOv5n loaded.")
        except Exception as e:
            print(f"[Server] YOLOv5 unavailable: {e}")
    return YOLO, YOLO_OK

def get_ocr():
    global OCR, OCR_OK
    if OCR is None:
        try:
            import easyocr
            OCR    = easyocr.Reader(['en'], gpu=False)
            OCR_OK = True
            print("[Server] EasyOCR loaded.")
        except Exception as e:
            print(f"[Server] EasyOCR unavailable: {e}")
    return OCR, OCR_OK

# ── OpenCV Face ───────────────────────────────────────────────────────────────
FACE_CASCADE  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Per-user recognizer cache: { username -> (recognizer, trained:bool) }
USER_RECOGNIZERS = {}
USER_REC_LOCK    = threading.Lock()

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
_db_conn = None
_db_lock = threading.Lock()

def get_db():
    global _db_conn
    if _db_conn is None:
        _db_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _db_conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                username     TEXT UNIQUE NOT NULL,
                created_at   TEXT
            )
        """)
        _db_conn.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                name        TEXT NOT NULL,
                description TEXT,
                first_seen  TEXT,
                last_seen   TEXT,
                meet_count  INTEGER DEFAULT 1,
                face_data   BLOB,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        _db_conn.commit()
    return _db_conn

def db_get_or_create_user(username):
    db  = get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with _db_lock:
        row = db.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        if row:
            return row[0]
        db.execute("INSERT INTO users (username, created_at) VALUES (?,?)", (username, now))
        db.commit()
        return db.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()[0]

def db_get_persons(user_id):
    db = get_db()
    with _db_lock:
        return db.execute(
            "SELECT id,name,description,last_seen,meet_count,face_data FROM persons WHERE user_id=?",
            (user_id,)
        ).fetchall()

def db_save_person(user_id, name, description, face_bytes):
    db  = get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with _db_lock:
        db.execute(
            "INSERT INTO persons (user_id,name,description,first_seen,last_seen,meet_count,face_data) VALUES(?,?,?,?,?,?,?)",
            (user_id, name, description, now, now, 1, face_bytes)
        )
        db.commit()

def db_update_person(person_id):
    db  = get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with _db_lock:
        db.execute(
            "UPDATE persons SET last_seen=?, meet_count=meet_count+1 WHERE id=?",
            (now, person_id)
        )
        db.commit()

def db_delete_person(user_id, name):
    db = get_db()
    with _db_lock:
        db.execute(
            "DELETE FROM persons WHERE user_id=? AND LOWER(name)=LOWER(?)",
            (user_id, name)
        )
        db.commit()

# ═══════════════════════════════════════════════════════════════════════════════
# FACE RECOGNITION (per-user LBPH)
# ═══════════════════════════════════════════════════════════════════════════════
def get_recognizer(user_id):
    with USER_REC_LOCK:
        if user_id not in USER_RECOGNIZERS:
            USER_RECOGNIZERS[user_id] = {
                "rec":     cv2.face.LBPHFaceRecognizer_create(),
                "trained": False
            }
        return USER_RECOGNIZERS[user_id]

def train_user_recognizer(user_id):
    rows = db_get_persons(user_id)
    rec_data = get_recognizer(user_id)
    faces, labels = [], []
    for row in rows:
        face_data = row[5]
        if face_data is None:
            continue
        arr = np.frombuffer(face_data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            faces.append(cv2.resize(img, (100, 100)))
            labels.append(row[0])
    if faces:
        rec_data["rec"].train(faces, np.array(labels))
        rec_data["trained"] = True
        print(f"[Face] User {user_id}: trained on {len(faces)} face(s).")
    else:
        rec_data["trained"] = False

def extract_face(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None, None
    x, y, w, h = faces[0]
    crop = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
    return crop, faces[0]

def face_to_bytes(face_gray):
    _, buf = cv2.imencode('.jpg', face_gray)
    return buf.tobytes()

def decode_image(b64):
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
    img_pil   = Image.open(BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def get_username():
    """Extract username from request JSON or header."""
    data     = request.get_json(silent=True) or {}
    username = data.get("username", "").strip().lower()
    if not username:
        username = request.headers.get("X-Username", "").strip().lower()
    return username or "default"

# ═══════════════════════════════════════════════════════════════════════════════
# OCR JOB QUEUE
# ═══════════════════════════════════════════════════════════════════════════════
OCR_JOBS = {}
OCR_LOCK = threading.Lock()

def run_ocr_job(job_id, frame):
    try:
        ocr, ocr_ok = get_ocr()
        if not ocr_ok:
            with OCR_LOCK:
                OCR_JOBS[job_id] = {"status": "error", "result": {"text": "", "message": "OCR not available."}}
            return
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = ocr.readtext(gray, detail=1, paragraph=True)
        texts   = []
        for r in results:
            try:
                if len(r) >= 3 and r[2] > 0.30:
                    texts.append(str(r[1]))
            except:
                pass
        full   = " ".join(texts).strip()
        result = {"text": full, "message": f"Text detected: {full}"} if full else \
                 {"text": "", "message": "No text found. Try holding camera closer to the text."}
        with OCR_LOCK:
            OCR_JOBS[job_id] = {"status": "done", "result": result}
    except Exception as e:
        with OCR_LOCK:
            OCR_JOBS[job_id] = {"status": "error", "result": {"text": "", "message": f"OCR error: {str(e)}"}}

# ═══════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)

# ── Serve HTML app ────────────────────────────────────────────────────────────
@app.route("/app")
def serve_app():
    html = Path(__file__).parent / "visionmate_app.html"
    if html.exists():
        return send_file(str(html), mimetype="text/html")
    return "visionmate_app.html not found.", 404

# ── Status ────────────────────────────────────────────────────────────────────
@app.route("/status")
def status():
    return jsonify({
        "status":  "ok",
        "yolo":    YOLO is not None,
        "ocr":     OCR is not None,
        "message": "VisionMate server is running."
    })

# ── Login / Register user ─────────────────────────────────────────────────────
@app.route("/login", methods=["POST"])
def login():
    data     = request.get_json()
    username = data.get("username", "").strip().lower()
    if not username:
        return jsonify({"error": "Username required."}), 400
    user_id = db_get_or_create_user(username)
    train_user_recognizer(user_id)
    persons = db_get_persons(user_id)
    return jsonify({
        "success":  True,
        "user_id":  user_id,
        "username": username,
        "persons":  len(persons),
        "message":  f"Welcome, {username.title()}! You have {len(persons)} person(s) in memory."
    })

# ── Get persons ───────────────────────────────────────────────────────────────
@app.route("/persons", methods=["POST"])
def get_persons():
    username = get_username()
    user_id  = db_get_or_create_user(username)
    rows     = db_get_persons(user_id)
    return jsonify({"persons": [{
        "id":          r[0],
        "name":        r[1],
        "description": r[2] or "",
        "last_seen":   r[3] or "Never",
        "meet_count":  r[4] or 1,
    } for r in rows]})

# ── Forget person ─────────────────────────────────────────────────────────────
@app.route("/forget", methods=["POST"])
def forget():
    data     = request.get_json()
    username = data.get("username", "").strip().lower()
    name     = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "No name provided."}), 400
    user_id = db_get_or_create_user(username)
    db_delete_person(user_id, name)
    train_user_recognizer(user_id)
    return jsonify({"message": f"{name} removed from your memory."})

# ── Register face ─────────────────────────────────────────────────────────────
@app.route("/register", methods=["POST"])
def register():
    data        = request.get_json()
    username    = data.get("username", "").strip().lower()
    b64         = data.get("image", "")
    name        = data.get("name", "").strip().title()
    description = data.get("description", "").strip()

    if not b64 or not name:
        return jsonify({"error": "image and name required."}), 400

    frame     = decode_image(b64)
    face_crop, _ = extract_face(frame)

    if face_crop is None:
        return jsonify({
            "success": False,
            "message": "No face detected. Ensure good lighting and face is clearly visible."
        })

    user_id = db_get_or_create_user(username)
    db_save_person(user_id, name, description, face_to_bytes(face_crop))
    train_user_recognizer(user_id)

    return jsonify({"success": True, "message": f"{name} registered successfully!"})

# ── Face recognition ──────────────────────────────────────────────────────────
@app.route("/face", methods=["POST"])
def face_endpoint():
    data     = request.get_json()
    username = data.get("username", "").strip().lower()
    b64      = data.get("image", "")

    if not b64:
        return jsonify({"error": "No image."}), 400

    frame     = decode_image(b64)
    face_crop, _ = extract_face(frame)

    if face_crop is None:
        return jsonify({
            "found":  False,
            "is_new": False,
            "message": "No face detected. Point camera at a person's face."
        })

    user_id  = db_get_or_create_user(username)
    rec_data = get_recognizer(user_id)

    if not rec_data["trained"]:
        return jsonify({
            "found":  False,
            "is_new": True,
            "message": "New person detected. Register them using the + button."
        })

    try:
        label, confidence = rec_data["rec"].predict(face_crop)
        if confidence < 80:
            rows = db_get_persons(user_id)
            person = next((r for r in rows if r[0] == label), None)
            if person:
                db_update_person(person[0])
                train_user_recognizer(user_id)
                name  = person[1]
                desc  = person[2] or ""
                seen  = person[3] or "earlier"
                count = person[4] or 1
                msg   = f"This is {name}."
                if desc: msg += f" {desc}."
                msg += f" You have met them {count} time{'s' if count!=1 else ''}. Last seen: {seen}."
                return jsonify({
                    "found":       True,
                    "is_new":      False,
                    "name":        name,
                    "description": desc,
                    "last_seen":   seen,
                    "meet_count":  count,
                    "confidence":  round((100-confidence)/100, 2),
                    "message":     msg
                })
    except Exception as e:
        print(f"[Face] Predict error: {e}")

    return jsonify({
        "found":  False,
        "is_new": True,
        "message": "New person detected. Register them using the + button."
    })

# ── OCR ───────────────────────────────────────────────────────────────────────
@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    data = request.get_json()
    b64  = data.get("image", "")
    if not b64:
        return jsonify({"error": "No image."}), 400

    frame  = decode_image(b64)
    job_id = str(uuid.uuid4())
    with OCR_LOCK:
        OCR_JOBS[job_id] = {"status": "pending", "result": None}
    threading.Thread(target=run_ocr_job, args=(job_id, frame), daemon=True).start()
    return jsonify({"job_id": job_id, "status": "pending", "message": "Processing..."})

@app.route("/ocr_result/<job_id>")
def ocr_result(job_id):
    with OCR_LOCK:
        job = OCR_JOBS.get(job_id)
    if not job:
        return jsonify({"status": "error", "message": "Job not found."}), 404
    if job["status"] == "pending":
        return jsonify({"status": "pending", "message": "Still processing..."})
    with OCR_LOCK:
        OCR_JOBS.pop(job_id, None)
    return jsonify({"status": "done", **job["result"]})

# ── Object detection ──────────────────────────────────────────────────────────
@app.route("/object", methods=["POST"])
def object_endpoint():
    data = request.get_json()
    b64  = data.get("image", "")
    if not b64:
        return jsonify({"error": "No image."}), 400

    yolo, yolo_ok = get_yolo()
    if not yolo_ok:
        return jsonify({"objects": [], "message": "Object detection not available."})

    frame   = decode_image(b64)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo(rgb, size=480)
    preds   = results.pandas().xyxy[0]
    labels  = sorted(set(preds["name"].tolist()))
    counts  = preds["name"].value_counts().to_dict()

    if labels:
        parts = [f"{counts[l]} {l}{'s' if counts[l]>1 else ''}" for l in labels]
        return jsonify({"objects": labels, "message": f"I can see: {', '.join(parts)}."})
    return jsonify({"objects": [], "message": "No objects detected. Try pointing at something closer."})

# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    get_db()  # Initialize DB
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"

    print("\n" + "="*55)
    print("  VisionMate Server Starting")
    print(f"  Local IP  : http://{local_ip}:5000")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
