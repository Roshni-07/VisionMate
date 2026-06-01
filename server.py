import os
import sqlite3
import base64
import requests
import time
import re
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# ── API Keys ──────────────────────────────────────────────────────────────────
FACEPP_KEY    = os.environ.get("FACEPP_KEY",    "OWFgrqZhO0P1N_tu4ITyRqFDVJXC43sz")
FACEPP_SECRET = os.environ.get("FACEPP_SECRET", "tdbRiyI11TKBEr9bPAvmnWEq0seGx1Tz")
OCRSPACE_KEY  = os.environ.get("OCRSPACE_KEY",  "K82034259688957")
IMAGGA_KEY    = os.environ.get("IMAGGA_KEY",    "")
IMAGGA_SECRET = os.environ.get("IMAGGA_SECRET", "")

DB_PATH = "visionmate.db"

# ── Database ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            faceset_token TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            face_token TEXT NOT NULL,
            description TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ── Helpers ───────────────────────────────────────────────────────────────────
def compress_image_b64(image_bytes, max_size=(800, 800), quality=75):
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail(max_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

def facepp_post(endpoint, extra_data=None, image_b64=None):
    data = {"api_key": FACEPP_KEY, "api_secret": FACEPP_SECRET}
    if extra_data:
        data.update(extra_data)
    if image_b64:
        data["image_base64"] = image_b64
    r = requests.post(f"https://api-us.faceplusplus.com{endpoint}", data=data, timeout=20)
    return r.json()

def ensure_faceset(user_id, user_name):
    db = get_db()
    row = db.execute("SELECT faceset_token FROM users WHERE id=?", (user_id,)).fetchone()
    db.close()
    if row and row["faceset_token"]:
        return row["faceset_token"]
    outer_id = f"user_{user_id}_{int(time.time())}"
    res = facepp_post("/facepp/v3/faceset/create", {"outer_id": outer_id, "display_name": user_name})
    token = res.get("faceset_token")
    if token:
        db = get_db()
        db.execute("UPDATE users SET faceset_token=? WHERE id=?", (token, user_id))
        db.commit()
        db.close()
    return token

def clean_ocr_text(raw):
    if not raw:
        return ""
    lines = raw.strip().splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        alnum = sum(c.isalnum() or c.isspace() for c in line)
        if len(line) > 0 and alnum / len(line) < 0.4:
            continue
        cleaned.append(line)
    return " ".join(cleaned)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/status")
def status():
    return jsonify({"status": "ok", "version": "2.0"})

@app.route("/app")
def serve_app():
    return send_file("visionmate_app.html")

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name required"}), 400
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE name=?", (name,)).fetchone()
    if not user:
        db.execute("INSERT INTO users (name) VALUES (?)", (name,))
        db.commit()
        user = db.execute("SELECT * FROM users WHERE name=?", (name,)).fetchone()
    user_id = user["id"]
    db.close()
    ensure_faceset(user_id, name)
    return jsonify({"user_id": user_id, "name": name})

# ── Face — MULTI-FACE ─────────────────────────────────────────────────────────
@app.route("/face", methods=["POST"])
def face():
    data = request.get_json()
    user_id   = data.get("user_id")
    image_b64 = data.get("image")
    if not image_b64 or not user_id:
        return jsonify({"error": "Missing data"}), 400
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    image_bytes = base64.b64decode(image_b64)
    image_b64c  = compress_image_b64(image_bytes)

    detect_res = facepp_post("/facepp/v3/detect", image_b64=image_b64c)
    faces = detect_res.get("faces", [])
    if not faces:
        return jsonify({"result": "No faces detected in the image.", "count": 0})

    db = get_db()
    faceset_row = db.execute("SELECT faceset_token FROM users WHERE id=?", (user_id,)).fetchone()
    db.close()
    faceset_token = faceset_row["faceset_token"] if faceset_row else None

    named = []
    unknown_count = 0

    for face_obj in faces:
        token = face_obj["face_token"]
        recognized = False
        if faceset_token:
            search_res = facepp_post(
                "/facepp/v3/search",
                {"faceset_token": faceset_token, "face_token": token}
            )
            results = search_res.get("results", [])
            if results and results[0].get("confidence", 0) >= 70:
                best = results[0]
                db = get_db()
                person = db.execute(
                    "SELECT name, description FROM persons WHERE face_token=? AND user_id=?",
                    (best["face_token"], user_id)
                ).fetchone()
                db.close()
                if person:
                    label = person["name"]
                    if person["description"]:
                        label += f" ({person['description']})"
                    named.append(label)
                    recognized = True
        if not recognized:
            unknown_count += 1

    parts = []
    if named:
        parts.append("This is " + ", ".join(named) + ".")
    if unknown_count == 1:
        parts.append("There is 1 unregistered face.")
    elif unknown_count > 1:
        parts.append(f"There are {unknown_count} unregistered faces.")

    result = " ".join(parts) if parts else "No recognized faces found."
    return jsonify({"result": result, "count": len(faces), "named": named, "unknown": unknown_count})

# ── OCR ───────────────────────────────────────────────────────────────────────
@app.route("/ocr", methods=["POST"])
def ocr():
    data = request.get_json()
    image_b64 = data.get("image", "")
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    resp = requests.post(
        "https://api.ocr.space/parse/image",
        data={
            "apikey": OCRSPACE_KEY,
            "base64Image": f"data:image/jpeg;base64,{image_b64}",
            "isOverlayRequired": False,
            "language": "eng",
            "isCreateSearchablePdf": False,
            "OCREngine": 2,
        },
        timeout=30
    )
    result = resp.json()
    parsed = result.get("ParsedResults", [])
    if not parsed:
        return jsonify({"result": "Could not read any text."})
    raw_text = parsed[0].get("ParsedText", "")
    cleaned = clean_ocr_text(raw_text)
    if not cleaned:
        return jsonify({"result": "No readable text found."})
    return jsonify({"result": cleaned})

# ── Object Detection — IMAGGA ─────────────────────────────────────────────────
@app.route("/object", methods=["POST"])
def object_detect():
    data = request.get_json()
    image_b64 = data.get("image", "")
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    if not IMAGGA_KEY or not IMAGGA_SECRET:
        return jsonify({"result": "Object detection not configured. Add IMAGGA_KEY and IMAGGA_SECRET in Render environment variables."})
    image_bytes = base64.b64decode(image_b64)
    image_b64c  = compress_image_b64(image_bytes)
    resp = requests.post(
        "https://api.imagga.com/v2/tags",
        auth=(IMAGGA_KEY, IMAGGA_SECRET),
        data={"image_base64": image_b64c},
        timeout=20
    )
    res = resp.json()
    tags = res.get("result", {}).get("tags", [])
    if not tags:
        return jsonify({"result": "Could not identify any objects."})

    # Raise threshold to 50 — eliminates weak/vague guesses
    filtered = [t for t in tags if t.get("confidence", 0) > 50]

    # Synonym groups — keep only first match per group to avoid repetition
    synonym_groups = [
        {"laptop", "computer", "notebook computer", "netbook", "macbook", "personal computer", "pc"},
        {"phone", "mobile phone", "smartphone", "cellphone", "mobile", "iphone", "android"},
        {"desk", "table", "desktop", "workspace", "workstation"},
        {"screen", "monitor", "display", "lcd", "led"},
        {"keyboard", "keypad"},
        {"person", "man", "woman", "human", "people", "individual", "adult"},
        {"chair", "seat", "furniture", "sofa", "couch"},
        {"book", "textbook", "notebook", "journal"},
        {"bag", "backpack", "handbag", "purse"},
        {"bottle", "water bottle", "flask", "container"},
        {"cup", "mug", "glass", "tumbler"},
        {"pen", "pencil", "marker", "stylus"},
        {"car", "vehicle", "automobile", "truck", "van"},
        {"food", "meal", "dish", "plate"},
    ]

    def get_group(label):
        l = label.lower()
        for g in synonym_groups:
            if l in g:
                return frozenset(g)
        return frozenset({l})

    seen_groups = set()
    deduped = []
    for t in filtered:
        label = t["tag"]["en"]
        grp = get_group(label)
        if grp not in seen_groups:
            seen_groups.add(grp)
            deduped.append(label)
        if len(deduped) == 3:
            break

    if not deduped:
        return jsonify({"result": "Could not identify any objects clearly."})

    result = "I can see: " + ", ".join(deduped) + "."
    return jsonify({"result": result, "tags": deduped})

# ── Register ──────────────────────────────────────────────────────────────────
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    user_id     = data.get("user_id")
    name        = data.get("name", "").strip()
    image_b64   = data.get("image", "")
    description = data.get("description", "").strip()
    if not all([user_id, name, image_b64]):
        return jsonify({"error": "Missing data"}), 400
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    image_bytes = base64.b64decode(image_b64)
    image_b64c  = compress_image_b64(image_bytes)
    detect_res = facepp_post("/facepp/v3/detect", image_b64=image_b64c)
    faces = detect_res.get("faces", [])
    if not faces:
        return jsonify({"error": "No face detected in image"}), 400
    face_token    = faces[0]["face_token"]
    faceset_token = ensure_faceset(user_id, name)
    facepp_post("/facepp/v3/faceset/addface", {
        "faceset_token": faceset_token,
        "face_tokens": face_token
    })
    db = get_db()
    existing = db.execute(
        "SELECT id FROM persons WHERE user_id=? AND name=?", (user_id, name)
    ).fetchone()
    if existing:
        db.execute(
            "UPDATE persons SET face_token=?, description=? WHERE id=?",
            (face_token, description, existing["id"])
        )
    else:
        db.execute(
            "INSERT INTO persons (user_id, name, face_token, description) VALUES (?,?,?,?)",
            (user_id, name, face_token, description)
        )
    db.commit()
    db.close()
    return jsonify({"result": f"{name} has been registered successfully."})

# ── Forget ────────────────────────────────────────────────────────────────────
@app.route("/forget", methods=["POST"])
def forget():
    data = request.get_json()
    user_id = data.get("user_id")
    name    = data.get("name", "").strip()
    if not user_id or not name:
        return jsonify({"error": "Missing data"}), 400
    db = get_db()
    person = db.execute(
        "SELECT face_token FROM persons WHERE user_id=? AND name=?", (user_id, name)
    ).fetchone()
    if not person:
        db.close()
        return jsonify({"error": f"{name} not found"}), 404
    face_token   = person["face_token"]
    faceset_row  = db.execute("SELECT faceset_token FROM users WHERE id=?", (user_id,)).fetchone()
    if faceset_row and faceset_row["faceset_token"]:
        facepp_post("/facepp/v3/faceset/removeface", {
            "faceset_token": faceset_row["faceset_token"],
            "face_tokens": face_token
        })
    db.execute("DELETE FROM persons WHERE user_id=? AND name=?", (user_id, name))
    db.commit()
    db.close()
    return jsonify({"result": f"{name} has been forgotten."})

# ── Persons List ──────────────────────────────────────────────────────────────
@app.route("/persons", methods=["POST"])
def persons():
    data = request.get_json()
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400
    db = get_db()
    rows = db.execute(
        "SELECT name, description FROM persons WHERE user_id=?", (user_id,)
    ).fetchall()
    db.close()
    return jsonify({"persons": [{"name": r["name"], "description": r["description"]} for r in rows]})

# ── Update Person Description ─────────────────────────────────────────────────
@app.route("/update_person", methods=["POST"])
def update_person():
    data        = request.get_json()
    user_id     = data.get("user_id")
    name        = data.get("name", "").strip()
    description = data.get("description", "").strip()
    if not user_id or not name:
        return jsonify({"error": "Missing data"}), 400
    db = get_db()
    result = db.execute(
        "UPDATE persons SET description=? WHERE user_id=? AND name=?",
        (description, user_id, name)
    )
    db.commit()
    db.close()
    if result.rowcount == 0:
        return jsonify({"error": f"{name} not found"}), 404
    return jsonify({"result": f"{name}'s details updated."})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
