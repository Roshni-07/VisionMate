import os
import sqlite3
import base64
import requests
import time
import re
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# ── API Keys — loaded from environment only, no hardcoded fallbacks ───────────
FACEPP_KEY    = os.environ.get("FACEPP_KEY",    "")
FACEPP_SECRET = os.environ.get("FACEPP_SECRET", "")
OCRSPACE_KEY  = os.environ.get("OCRSPACE_KEY",  "")
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
    # NEW: activity log table — tracks every scan event
    c.execute("""
        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            detail TEXT,
            timestamp TEXT NOT NULL,
            location TEXT
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
def log_activity(user_id, action, detail="", location=""):
    """Log an activity event to the database."""
    try:
        db = get_db()
        db.execute(
            "INSERT INTO activity_log (user_id, action, detail, timestamp, location) VALUES (?,?,?,?,?)",
            (user_id, action, detail, datetime.now().isoformat(), location)
        )
        db.commit()
        db.close()
    except Exception:
        pass  # Never crash a scan just because logging failed

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
    return jsonify({"status": "ok", "version": "3.0"})

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
    log_activity(user_id, "login", name)
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

    # Log scan
    log_activity(user_id, "face_scan", result)

    return jsonify({"result": result, "count": len(faces), "named": named, "unknown": unknown_count})

# ── OCR ───────────────────────────────────────────────────────────────────────
@app.route("/ocr", methods=["POST"])
def ocr():
    data = request.get_json()
    user_id   = data.get("user_id")
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

    # Log scan
    log_activity(user_id, "ocr_scan", cleaned[:100])

    return jsonify({"result": cleaned})

# ── Object Detection — IMAGGA ─────────────────────────────────────────────────
@app.route("/object", methods=["POST"])
def object_detect():
    data = request.get_json()
    user_id   = data.get("user_id")
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

    filtered = [t for t in tags if t.get("confidence", 0) > 30]

    # canonical label → all synonyms Imagga may return for same object
    SYNONYM_GROUPS = {
        # Computing
        "laptop":           {"laptop", "computer", "notebook computer", "netbook", "macbook", "personal computer", "pc", "chromebook", "ultrabook"},
        "phone":            {"phone", "mobile phone", "smartphone", "cellphone", "mobile", "iphone", "android", "handset", "telephone"},
        "tablet":           {"tablet", "ipad", "tablet computer", "e-reader", "kindle"},
        "keyboard":         {"keyboard", "keypad", "numeric keypad"},
        "mouse":            {"mouse", "computer mouse", "trackpad", "touchpad"},
        "screen":           {"screen", "monitor", "display", "lcd", "led", "oled", "television", "tv", "smart tv"},
        "projector":        {"projector", "video projector", "beamer", "overhead projector", "dlp projector"},
        "printer":          {"printer", "laser printer", "inkjet printer", "scanner", "photocopier", "copier"},
        "router":           {"router", "modem", "wifi router", "network device", "access point"},
        "hard drive":       {"hard drive", "hard disk", "hdd", "ssd", "usb drive", "flash drive", "pendrive", "memory stick"},
        "camera":           {"camera", "webcam", "dslr", "digital camera", "video camera", "camcorder", "action camera"},
        "headphones":       {"headphones", "earphones", "earbuds", "headset", "airpods", "earplugs"},
        "speaker":          {"speaker", "loudspeaker", "bluetooth speaker", "sound system", "subwoofer"},
        "remote control":   {"remote control", "remote", "tv remote", "clicker"},
        # Furniture & room
        "table":            {"desk", "table", "dining table", "coffee table", "workspace", "workstation", "countertop", "counter"},
        "chair":            {"chair", "seat", "office chair", "sofa", "couch", "armchair", "bench", "stool", "recliner", "settee"},
        "bed":              {"bed", "mattress", "bunk bed", "cot", "crib", "bedframe"},
        "shelf":            {"shelf", "bookshelf", "bookcase", "rack", "cabinet", "cupboard", "wardrobe", "drawer", "dresser", "almira"},
        "door":             {"door", "doorway", "gate", "entrance", "exit", "sliding door"},
        "window":           {"window", "glass window", "window pane"},
        "whiteboard":       {"whiteboard", "blackboard", "chalkboard", "board", "presentation board", "smartboard"},
        "clock":            {"clock", "wall clock", "alarm clock", "watch", "wristwatch", "timer"},
        "lamp":             {"lamp", "light", "bulb", "tube light", "led light", "ceiling light", "desk lamp", "torch", "flashlight", "lantern"},
        "fan":              {"fan", "ceiling fan", "table fan", "pedestal fan", "exhaust fan"},
        "air conditioner":  {"air conditioner", "ac", "air conditioning", "split ac", "window ac", "hvac"},
        "heater":           {"heater", "room heater", "radiator", "electric heater"},
        "mirror":           {"mirror", "looking glass", "dressing mirror"},
        "curtain":          {"curtain", "drape", "blind", "shutter", "shade"},
        "carpet":           {"carpet", "rug", "mat", "floor mat", "doormat"},
        # Stationery & office
        "pen":              {"pen", "pencil", "marker", "stylus", "highlighter", "ballpoint", "fountain pen"},
        "book":             {"book", "textbook", "notebook", "journal", "magazine", "newspaper", "manual", "binder", "folder"},
        "paper":            {"paper", "document", "sheet", "form", "receipt", "bill", "invoice", "printout"},
        "scissors":         {"scissors", "cutter", "blade", "knife", "box cutter"},
        "stapler":          {"stapler", "staple", "hole punch"},
        "calculator":       {"calculator", "scientific calculator"},
        # Kitchen & food
        "cup":              {"cup", "mug", "glass", "tumbler", "teacup", "coffee cup"},
        "bottle":           {"bottle", "water bottle", "flask", "thermos", "container", "jar", "jug"},
        "plate":            {"plate", "dish", "bowl", "tray", "platter"},
        "food":             {"food", "meal", "snack", "lunch", "dinner", "breakfast", "fruit", "vegetable", "bread", "rice"},
        "spoon":            {"spoon", "fork", "knife", "cutlery", "spatula", "ladle"},
        "microwave":        {"microwave", "oven", "toaster", "microwave oven"},
        "refrigerator":     {"refrigerator", "fridge", "freezer"},
        # Clothing & personal
        "bag":              {"bag", "backpack", "handbag", "purse", "suitcase", "luggage", "briefcase", "tote bag"},
        "glasses":          {"glasses", "spectacles", "sunglasses", "eyeglasses", "goggles"},
        "helmet":           {"helmet", "hard hat", "safety helmet", "cap", "hat"},
        "shoes":            {"shoes", "sneakers", "sandals", "boots", "footwear", "slippers", "chappal"},
        "umbrella":         {"umbrella", "raincoat", "poncho"},
        # Medical & safety
        "medicine":         {"medicine", "tablet", "capsule", "pill", "syrup", "injection", "drug", "medication"},
        "wheelchair":       {"wheelchair", "walker", "crutches", "cane", "walking stick"},
        "fire extinguisher":{"fire extinguisher", "extinguisher", "fire safety"},
        "first aid":        {"first aid", "first aid kit", "bandage", "plaster", "medical kit"},
        # Transport & outdoor
        "car":              {"car", "vehicle", "automobile", "truck", "van", "jeep", "suv", "taxi", "cab"},
        "bicycle":          {"bicycle", "bike", "cycle", "scooter", "motorbike", "motorcycle"},
        "bus":              {"bus", "minibus", "coach", "school bus"},
        # People
        "person":           {"person", "man", "woman", "human", "people", "individual", "adult", "child", "kid", "student", "teacher"},
    }

    # Generic/abstract tags — useless to announce
    USELESS_TAGS = {
        "indoor", "outdoor", "technology", "wood", "wall", "floor",
        "ceiling", "background", "object", "item", "material", "no person",
        "surface", "texture", "color", "colour", "equipment", "still life",
        "electronic", "electronics", "device", "interior", "room", "space",
        "design", "pattern", "shape", "style", "modern", "old", "new",
        "dark", "light", "bright", "blur", "closeup", "close-up",
        "horizontal", "vertical", "nobody", "group", "collection",
        "various", "many", "single", "multiple", "set", "type"
    }

    def get_canonical(label):
        l = label.lower()
        for canonical, synonyms in SYNONYM_GROUPS.items():
            if l in synonyms:
                return canonical
        return label  # not in any group — use as-is

    seen = set()
    deduped = []
    for t in filtered:
        label = t["tag"]["en"]
        if label.lower() in USELESS_TAGS:
            continue
        canonical = get_canonical(label)
        if canonical not in seen:
            seen.add(canonical)
            deduped.append(canonical)
        if len(deduped) == 3:
            break

    if not deduped:
        return jsonify({"result": "Could not identify any objects clearly."})

    result = "I can see: " + ", ".join(deduped) + "."

    # Log scan
    log_activity(user_id, "object_scan", result)

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
    log_activity(user_id, "register", name)
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
    log_activity(user_id, "forget", name)
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
    log_activity(user_id, "update_person", f"{name}: {description}")
    return jsonify({"result": f"{name}'s details updated."})

# ── SOS ───────────────────────────────────────────────────────────────────────
@app.route("/sos", methods=["POST"])
def sos():
    data     = request.get_json()
    user_id  = data.get("user_id")
    location = data.get("location", "")
    db = get_db()
    user = db.execute("SELECT name FROM users WHERE id=?", (user_id,)).fetchone()
    db.close()
    user_name = user["name"] if user else "Unknown user"
    log_activity(user_id, "SOS", f"Emergency triggered by {user_name}", location)
    return jsonify({"status": "logged", "caregiver_phone": CAREGIVER_PHONE})

# ── Caregiver Dashboard ───────────────────────────────────────────────────────
@app.route("/caregiver", methods=["POST"])
def caregiver():
    data    = request.get_json()
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400
    db = get_db()
    logs = db.execute(
        "SELECT action, detail, timestamp, location FROM activity_log WHERE user_id=? ORDER BY timestamp DESC LIMIT 50",
        (user_id,)
    ).fetchall()
    user = db.execute("SELECT name FROM users WHERE id=?", (user_id,)).fetchone()
    db.close()
    return jsonify({
        "user_name": user["name"] if user else "Unknown",
        "logs": [{"action": r["action"], "detail": r["detail"], "timestamp": r["timestamp"], "location": r["location"]} for r in logs]
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
