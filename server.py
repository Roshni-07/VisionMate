"""
VisionMate – Lightweight API Server
Uses Google Vision API (OCR + Objects) and Face++ (Face Recognition)
Runs on Render free tier — no heavy models, fast responses
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sqlite3
import numpy as np
import json
import base64
import threading
import uuid
import os
import requests
from datetime import datetime
from pathlib import Path
from io import BytesIO
from PIL import Image
import socket

# ── API KEYS ──────────────────────────────────────────────────────────────────
GOOGLE_VISION_KEY = os.environ.get("GOOGLE_VISION_KEY", "AIzaSyDbQjiFP8ZtbxvtwgyC7SHvcJIc6KJTND0")
FACEPP_KEY        = os.environ.get("FACEPP_KEY",         "OWFgrqZhO0P1N_tu4ITyRqFDVJXC43sz")
FACEPP_SECRET     = os.environ.get("FACEPP_SECRET",      "tdbRiyI11TKBEr9bPAvmnWEq0seGx1Tz")

FACEPP_DETECT  = "https://api-us.faceplusplus.com/facepp/v3/detect"
FACEPP_FACESET = "https://api-us.faceplusplus.com/facepp/v3/faceset"
FACEPP_SEARCH  = "https://api-us.faceplusplus.com/facepp/v3/search"
VISION_URL     = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_KEY}"

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
DB_PATH = Path(os.environ.get("DB_PATH", "visionmate.db"))
DB_LOCK = threading.Lock()
_db_conn = None

def get_db():
    global _db_conn
    if _db_conn is None:
        _db_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _db_conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            faceset_token TEXT)""")
        _db_conn.execute("""CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            face_token TEXT,
            first_seen TEXT,
            last_seen TEXT,
            meet_count INTEGER DEFAULT 1)""")
        _db_conn.commit()
    return _db_conn

def db_get_user(username):
    with DB_LOCK:
        return get_db().execute(
            "SELECT id, username, faceset_token FROM users WHERE username=?",
            (username.lower(),)).fetchone()

def db_create_user(username):
    with DB_LOCK:
        get_db().execute("INSERT OR IGNORE INTO users (username) VALUES (?)", (username.lower(),))
        get_db().commit()
    return db_get_user(username)

def db_set_faceset(user_id, token):
    with DB_LOCK:
        get_db().execute("UPDATE users SET faceset_token=? WHERE id=?", (token, user_id))
        get_db().commit()

def db_get_persons(user_id):
    with DB_LOCK:
        return get_db().execute(
            "SELECT id,name,description,face_token,last_seen,meet_count FROM persons WHERE user_id=?",
            (user_id,)).fetchall()

def db_save_person(user_id, name, description, face_token):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with DB_LOCK:
        get_db().execute(
            "INSERT INTO persons (user_id,name,description,face_token,first_seen,last_seen,meet_count) VALUES(?,?,?,?,?,?,1)",
            (user_id, name, description, face_token, now, now))
        get_db().commit()

def db_update_person(person_id):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with DB_LOCK:
        get_db().execute(
            "UPDATE persons SET last_seen=?, meet_count=meet_count+1 WHERE id=?", (now, person_id))
        get_db().commit()

def db_delete_person(user_id, name):
    with DB_LOCK:
        get_db().execute(
            "DELETE FROM persons WHERE user_id=? AND LOWER(name)=LOWER(?)", (user_id, name))
        get_db().commit()

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def decode_b64(b64):
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)

def b64_to_jpeg(b64, quality=88):
    raw = decode_b64(b64)
    img = Image.open(BytesIO(raw)).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def clean_b64(b64):
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return b64

# ═══════════════════════════════════════════════════════════════════════════════
# FACE++ HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def facepp_detect(b64):
    try:
        jpeg = b64_to_jpeg(b64)
        resp = requests.post(FACEPP_DETECT, data={
            "api_key": FACEPP_KEY, "api_secret": FACEPP_SECRET,
        }, files={"image_file": ("face.jpg", jpeg, "image/jpeg")}, timeout=15)
        faces = resp.json().get("faces", [])
        return faces[0]["face_token"] if faces else None
    except Exception as e:
        print(f"[Face++] Detect: {e}")
        return None

def facepp_get_or_create_faceset(user_id, faceset_token):
    if faceset_token:
        return faceset_token
    try:
        resp = requests.post(f"{FACEPP_FACESET}/create", data={
            "api_key": FACEPP_KEY, "api_secret": FACEPP_SECRET,
            "display_name": f"user_{user_id}",
            "outer_id": f"visionmate_user_{user_id}",
        }, timeout=15)
        token = resp.json().get("faceset_token")
        if token:
            db_set_faceset(user_id, token)
        return token
    except Exception as e:
        print(f"[Face++] Faceset: {e}")
        return None

def facepp_add_face(faceset_token, face_token):
    try:
        requests.post(f"{FACEPP_FACESET}/addface", data={
            "api_key": FACEPP_KEY, "api_secret": FACEPP_SECRET,
            "faceset_token": faceset_token, "face_tokens": face_token,
        }, timeout=15)
    except Exception as e:
        print(f"[Face++] AddFace: {e}")

def facepp_search(faceset_token, b64):
    try:
        jpeg = b64_to_jpeg(b64)
        resp = requests.post(FACEPP_SEARCH, data={
            "api_key": FACEPP_KEY, "api_secret": FACEPP_SECRET,
            "faceset_token": faceset_token,
        }, files={"image_file": ("face.jpg", jpeg, "image/jpeg")}, timeout=15)
        results = resp.json().get("results", [])
        if results:
            return results[0]["face_token"], results[0]["confidence"]
        return None, 0
    except Exception as e:
        print(f"[Face++] Search: {e}")
        return None, 0

# ═══════════════════════════════════════════════════════════════════════════════
# GOOGLE VISION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def vision_ocr(b64):
    try:
        payload = {"requests": [{"image": {"content": clean_b64(b64)},
            "features": [{"type": "TEXT_DETECTION", "maxResults": 1}]}]}
        resp = requests.post(VISION_URL, json=payload, timeout=15)
        annotation = resp.json()["responses"][0].get("fullTextAnnotation")
        if annotation:
            return annotation["text"].strip().replace("\n", " ")
        return ""
    except Exception as e:
        print(f"[Vision] OCR: {e}")
        return ""

def vision_objects(b64):
    try:
        payload = {"requests": [{"image": {"content": clean_b64(b64)},
            "features": [
                {"type": "OBJECT_LOCALIZATION", "maxResults": 10},
                {"type": "LABEL_DETECTION", "maxResults": 5}
            ]}]}
        resp = requests.post(VISION_URL, json=payload, timeout=15)
        response = resp.json()["responses"][0]
        objects = [o["name"] for o in response.get("localizedObjectAnnotations", []) if o["score"] > 0.5]
        if not objects:
            objects = [l["description"] for l in response.get("labelAnnotations", []) if l["score"] > 0.7]
        return list(dict.fromkeys(objects))
    except Exception as e:
        print(f"[Vision] Objects: {e}")
        return []

# ═══════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)

@app.route("/app")
def serve_app():
    html = Path(__file__).parent / "visionmate_app.html"
    return send_file(str(html), mimetype="text/html") if html.exists() else ("App not found.", 404)

@app.route("/status")
def status():
    get_db()
    return jsonify({"status": "ok", "message": "VisionMate server is running.", "ocr": True, "yolo": True, "face": True})

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username", "").strip().lower()
    if not username:
        return jsonify({"success": False, "message": "Username required."})
    user = db_get_user(username) or db_create_user(username)
    persons = db_get_persons(user[0])
    return jsonify({"success": True, "user_id": user[0], "username": username,
        "message": f"Welcome, {username.title()}! {len(persons)} person(s) in your memory."})

@app.route("/persons", methods=["POST"])
def get_persons():
    data = request.get_json()
    user = db_get_user(data.get("username", "").strip().lower())
    if not user:
        return jsonify({"persons": []})
    return jsonify({"persons": [{"id": r[0], "name": r[1], "description": r[2] or "",
        "last_seen": r[4] or "Never", "meet_count": r[5] or 1} for r in db_get_persons(user[0])]})

@app.route("/forget", methods=["POST"])
def forget():
    data = request.get_json()
    user = db_get_user(data.get("username", "").strip().lower())
    name = data.get("name", "").strip()
    if user:
        db_delete_person(user[0], name)
    return jsonify({"message": f"{name} removed from memory."})

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username", "").strip().lower()
    b64      = data.get("image", "")
    name     = data.get("name", "").strip().title()
    desc     = data.get("description", "").strip()

    if not b64 or not name:
        return jsonify({"success": False, "message": "Image and name required."})

    user = db_get_user(username)
    if not user:
        return jsonify({"success": False, "message": "Please login again."})

    user_id, _, faceset_token = user
    face_token = facepp_detect(b64)
    if not face_token:
        return jsonify({"success": False,
            "message": "No face detected. Make sure the face is clearly visible and well lit."})

    faceset_token = facepp_get_or_create_faceset(user_id, faceset_token)
    if not faceset_token:
        return jsonify({"success": False, "message": "Could not create face database. Try again."})

    facepp_add_face(faceset_token, face_token)
    db_save_person(user_id, name, desc, face_token)
    return jsonify({"success": True, "message": f"{name} registered successfully!"})

@app.route("/face", methods=["POST"])
def face_endpoint():
    data = request.get_json()
    username = data.get("username", "").strip().lower()
    b64 = data.get("image", "")
    if not b64:
        return jsonify({"error": "No image."}), 400

    user = db_get_user(username)
    if not user:
        return jsonify({"found": False, "is_new": False, "message": "Please login again."})

    user_id, _, faceset_token = user
    persons = db_get_persons(user_id)

    if not persons or not faceset_token:
        face_token = facepp_detect(b64)
        if face_token:
            return jsonify({"found": False, "is_new": True,
                "message": "New person detected. Please register them using the Register button."})
        return jsonify({"found": False, "is_new": False,
            "message": "No face detected. Point the camera clearly at a person's face."})

    matched_token, confidence = facepp_search(faceset_token, b64)
    THRESHOLD = 70

    if matched_token and confidence >= THRESHOLD:
        for person in persons:
            if person[3] == matched_token:
                db_update_person(person[0])
                name  = person[1]
                desc  = person[2] or ""
                seen  = person[4] or "earlier"
                count = (person[5] or 1) + 1
                msg   = f"This is {name}."
                if desc: msg += f" {desc}."
                msg += f" You have met them {count} times. Last seen: {seen}."
                return jsonify({"found": True, "is_new": False, "name": name,
                    "description": desc, "last_seen": seen, "meet_count": count,
                    "confidence": round(confidence/100, 2), "message": msg})

    face_token = facepp_detect(b64)
    if face_token:
        return jsonify({"found": False, "is_new": True,
            "message": "New person detected. Please register them."})
    return jsonify({"found": False, "is_new": False,
        "message": "No face detected. Point camera at a face."})

# ── OCR (job-based so no timeout) ─────────────────────────────────────────────
OCR_JOBS = {}
OCR_LOCK_J = threading.Lock()

def run_ocr_job(job_id, b64):
    try:
        text = vision_ocr(b64)
        result = ({"text": text, "message": f"Text detected: {text}"} if text
                  else {"text": "", "message": "No text found. Try holding the camera closer."})
        with OCR_LOCK_J:
            OCR_JOBS[job_id] = {"status": "done", "result": result}
    except Exception as e:
        with OCR_LOCK_J:
            OCR_JOBS[job_id] = {"status": "error", "result": {"text": "", "message": str(e)}}

@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    data = request.get_json()
    b64  = data.get("image", "")
    if not b64:
        return jsonify({"error": "No image."}), 400
    job_id = str(uuid.uuid4())
    with OCR_LOCK_J:
        OCR_JOBS[job_id] = {"status": "pending", "result": None}
    threading.Thread(target=run_ocr_job, args=(job_id, b64), daemon=True).start()
    return jsonify({"job_id": job_id, "status": "pending"})

@app.route("/ocr_result/<job_id>")
def ocr_result(job_id):
    with OCR_LOCK_J:
        job = OCR_JOBS.get(job_id)
    if not job:
        return jsonify({"status": "error", "message": "Job not found."}), 404
    if job["status"] == "pending":
        return jsonify({"status": "pending", "message": "Processing..."})
    with OCR_LOCK_J:
        OCR_JOBS.pop(job_id, None)
    return jsonify({"status": "done", **job["result"]})

@app.route("/object", methods=["POST"])
def object_endpoint():
    data = request.get_json()
    b64  = data.get("image", "")
    if not b64:
        return jsonify({"error": "No image."}), 400
    labels = vision_objects(b64)
    if labels:
        return jsonify({"objects": labels, "message": f"I can see: {', '.join(labels[:6])}."})
    return jsonify({"objects": [], "message": "No objects detected. Try pointing at something closer."})

# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    get_db()
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except:
        local_ip = "127.0.0.1"
    print(f"\n{'='*50}\n  VisionMate Server (Lightweight API Edition)\n  http://{local_ip}:5000\n{'='*50}\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
