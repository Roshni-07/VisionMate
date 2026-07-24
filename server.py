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
CAREGIVER_PHONE = os.environ.get("CAREGIVER_PHONE", "")


DB_PATH = os.environ.get("DB_PATH", "visionmate.db")

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
    try:
        db = get_db()
        db.execute(
            "INSERT INTO activity_log (user_id, action, detail, timestamp, location) VALUES (?,?,?,?,?)",
            (user_id, action, detail, datetime.now().isoformat(), location)
        )
        db.commit()
        db.close()
    except Exception:
        pass

def compress_image_b64(image_bytes, max_size=(800, 800), quality=75):
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail(max_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

def compress_image_b64_object(image_bytes, max_size=(1024, 1024), quality=85):
    """Higher-res than default compress — Imagga tags improve with more detail."""
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail(max_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

def compress_image_b64_ocr(image_bytes):
    from PIL import ImageEnhance, ImageFilter
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.SHARPEN)
    img.thumbnail((1200, 1200), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def facepp_post(endpoint, extra_data=None, image_b64=None):
    data = {"api_key": FACEPP_KEY, "api_secret": FACEPP_SECRET}
    if extra_data:
        data.update(extra_data)
    if image_b64:
        data["image_base64"] = image_b64
    try:
        r = requests.post(f"https://api-us.faceplusplus.com{endpoint}", data=data, timeout=20)
        return r.json()
    except requests.exceptions.RequestException:
        return {"error_message": "NETWORK_ERROR"}

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
    return jsonify({"status": "ok", "version": "4.2"})

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

# ── Face — MULTI-FACE with proximity sort ─────────────────────────────────────
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
    if detect_res.get("error_message"):
        return jsonify({"result": "Face detection service unavailable right now. Try again.", "count": 0})
    faces = detect_res.get("faces", [])
    if not faces:
        return jsonify({"result": "No faces detected in the image.", "count": 0})

    faces.sort(
        key=lambda f: f["face_rectangle"]["width"] * f["face_rectangle"]["height"],
        reverse=True
    )
    faces = faces[:3]

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
            if search_res.get("error_message"):
                unknown_count += 1
                continue
            results = search_res.get("results", [])
            if results and results[0].get("confidence", 0) >= 80:
                best = results[0]
                confidence = round(results[0]["confidence"], 1)
                db = get_db()
                matched_token = best["face_token"]
                person = db.execute(
                    """SELECT name, description FROM persons
                       WHERE user_id=? AND (
                           face_token = ? OR
                           face_token LIKE ? OR
                           face_token LIKE ? OR
                           face_token LIKE ?
                       )""",
                    (user_id, matched_token,
                     matched_token + ",%",
                     "%," + matched_token,
                     "%," + matched_token + ",%")
                ).fetchone()
                db.close()
                if person:
                    label = person["name"]
                    if person["description"]:
                        label += f" ({person['description']})"
                    named.append({"name": label, "confidence": confidence, "borderline": confidence < 85})
                    recognized = True
        if not recognized:
            unknown_count += 1

    parts = []
    if named:
        seen_names = set()
        confident, borderline = [], []
        for n in named:
            if n["name"] in seen_names:
                continue
            seen_names.add(n["name"])
            (borderline if n["borderline"] else confident).append(n["name"])
        if confident:
            parts.append("I can see " + ", ".join(confident) + ".")
        if borderline:
            parts.append("Possibly " + ", ".join(borderline) + " — not fully certain.")
    if unknown_count == 1:
        parts.append("There is 1 unregistered face.")
    elif unknown_count > 1:
        parts.append(f"There are {unknown_count} unregistered faces.")

    result = " ".join(parts) if parts else "No recognized faces found."

    log_detail = ", ".join(f"{n['name']} ({n['confidence']}%)" for n in named) if named else result
    log_activity(user_id, "face_scan", log_detail)

    return jsonify({
        "result": result,
        "count": len(faces),
        "named": [n["name"] for n in named],
        "unknown": unknown_count
    })

# ── OCR ───────────────────────────────────────────────────────────────────────
@app.route("/ocr", methods=["POST"])
def ocr():
    data = request.get_json()
    user_id   = data.get("user_id")
    image_b64 = data.get("image", "")
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    if not OCRSPACE_KEY:
        return jsonify({"result": "Text reading not configured. Add OCRSPACE_KEY in Render environment variables."})
    image_bytes = base64.b64decode(image_b64)
    image_b64c  = compress_image_b64_ocr(image_bytes)
    try:
        resp = requests.post(
            "https://api.ocr.space/parse/image",
            data={
                "apikey": OCRSPACE_KEY,
                "base64Image": f"data:image/jpeg;base64,{image_b64c}",
                "isOverlayRequired": False,
                "language": "eng",
                "isCreateSearchablePdf": False,
                "OCREngine": 2,
            },
            timeout=30
        )
        result = resp.json()
    except requests.exceptions.RequestException:
        return jsonify({"result": "Text reading service unavailable right now. Try again."})
    if result.get("IsErroredOnProcessing"):
        return jsonify({"result": "Could not read any text."})
    parsed = result.get("ParsedResults", [])
    if not parsed:
        return jsonify({"result": "Could not read any text."})
    raw_text = parsed[0].get("ParsedText", "")
    cleaned = clean_ocr_text(raw_text)
    if not cleaned:
        return jsonify({"result": "No readable text found."})

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
    image_b64c  = compress_image_b64_object(image_bytes)
    try:
        resp = requests.post(
            "https://api.imagga.com/v2/tags",
            auth=(IMAGGA_KEY, IMAGGA_SECRET),
            data={"image_base64": image_b64c, "limit": 30, "language": "en"},
            timeout=20
        )
        res = resp.json()
    except requests.exceptions.RequestException:
        return jsonify({"result": "Object detection service unavailable right now. Try again."})
    tags = res.get("result", {}).get("tags", [])
    if not tags:
        return jsonify({"result": "Could not identify any objects."})

    # FIX: raised floor 30 -> 45, plus relative-margin filter vs top tag
    # kills low-confidence noise (pen/finger on a slipper) that passed before
    filtered = [t for t in tags if t.get("confidence", 0) > 20]
    if filtered:
        top_conf = filtered[0].get("confidence", 0)
        filtered = [t for t in filtered if t.get("confidence", 0) >= top_conf * 0.5]

    USELESS_TAGS = {
        "image", "photography", "photo", "picture", "color", "colour", "design",
        "art", "illustration", "graphic", "texture", "pattern", "background",
        "shape", "object", "thing", "item", "element", "material", "surface",
        "light", "dark", "white", "black", "indoor", "outdoor", "nobody",
        "no person", "single", "isolated", "closeup", "close-up", "style",
        # generic scene/room descriptors — crowd out real object tags
        "interior", "room", "home", "house", "equipment", "device",
        "furniture", "wood", "wooden", "stuff", "workspace interior",
        # body parts — creep in when object held close to camera
        "finger", "fingers", "hand", "hands", "skin", "nail", "nails",
        "thumb", "palm", "arm", "wrist", "knuckle"
    }

    SYNONYM_GROUPS = [
        ({"laptop","computer","notebook computer","netbook","macbook","personal computer","pc"}, "laptop"),
        ({"phone","mobile phone","smartphone","cellphone","mobile","iphone","android"}, "phone"),
        ({"tablet","ipad","tab"}, "tablet"),
        ({"keyboard","keypad"}, "keyboard"),
        ({"mouse","computer mouse"}, "mouse"),
        ({"screen","monitor","display","lcd","led","television","tv"}, "screen"),
        ({"projector","overhead projector"}, "projector"),
        ({"printer"}, "printer"),
        ({"router","modem","wifi router"}, "router"),
        ({"hard drive","hard disk","external drive","usb drive","pendrive","flash drive"}, "hard drive"),
        ({"camera","webcam","dslr","digital camera"}, "camera"),
        ({"headphones","headset","earphones","earbuds","airpods"}, "headphones"),
        ({"speaker","loudspeaker","bluetooth speaker"}, "speaker"),
        ({"remote control","remote"}, "remote control"),
        ({"desk","table","desktop","workspace","workstation","dining table"}, "table"),
        ({"chair","seat","sofa","couch","armchair","bench"}, "chair"),
        ({"bed","mattress","cot"}, "bed"),
        ({"shelf","bookshelf","rack","cabinet","cupboard","drawer","wardrobe"}, "shelf"),
        ({"door","doorway","entrance","gate"}, "door"),
        ({"window","windowpane"}, "window"),
        ({"whiteboard","blackboard","chalkboard","board"}, "whiteboard"),
        ({"clock","watch","wristwatch","alarm clock","wall clock"}, "clock"),
        ({"lamp","light","bulb","torch","flashlight","tube light"}, "lamp"),
        ({"fan","ceiling fan","table fan"}, "fan"),
        ({"air conditioner","ac","air conditioning"}, "air conditioner"),
        ({"heater","radiator","room heater"}, "heater"),
        ({"mirror"}, "mirror"),
        ({"curtain","blinds","drape"}, "curtain"),
        ({"carpet","rug","mat","floor mat"}, "carpet"),
        ({"pen","pencil","marker","stylus","crayon"}, "pen"),
        ({"book","textbook","notebook","journal","magazine","comic"}, "book"),
        ({"paper","document","sheet","newspaper"}, "paper"),
        ({"scissors","cutter"}, "scissors"),
        ({"stapler"}, "stapler"),
        ({"calculator"}, "calculator"),
        ({"cup","mug","glass","tumbler","teacup"}, "cup"),
        ({"bottle","water bottle","flask","container","jar","thermos","vacuum flask","insulated bottle"}, "bottle"),
        ({"plate","dish","tray","bowl"}, "plate"),
        ({"food","meal","snack","fruit","vegetable","bread","rice"}, "food"),
        ({"spoon","fork","knife","cutlery","chopsticks"}, "spoon"),
        ({"microwave","oven","microwave oven"}, "microwave"),
        ({"refrigerator","fridge","freezer"}, "refrigerator"),
        ({"bag","backpack","handbag","suitcase","luggage","pouch","zipper pouch"}, "bag"),
        ({"id card","identity card","badge","lanyard","access card","name tag"}, "ID card"),
        ({"wallet","billfold","purse","pocketbook","money clip","card holder"}, "wallet"),
        ({"stuffed animal","stuffed toy","soft toy","plush toy","plush","teddy bear","teddy","doll","toy"}, "soft toy"),
        ({"blanket","quilt","throw blanket","comforter","duvet","textile","fabric","bedspread"}, "blanket"),
        ({"glasses","spectacles","sunglasses","eyeglasses"}, "glasses"),
        ({"helmet","hard hat"}, "helmet"),
        ({"shoes","sneakers","sandals","boots","footwear","slipper","slippers","flip flop","flip-flop","flip flops","sandal"}, "footwear"),
        ({"umbrella","parasol"}, "umbrella"),
        ({"medicine","pill","tablet","capsule","syringe","injection"}, "medicine"),
        ({"wheelchair"}, "wheelchair"),
        ({"fire extinguisher"}, "fire extinguisher"),
        ({"first aid","first aid kit","bandage"}, "first aid"),
        ({"car","vehicle","automobile","truck","van","suv"}, "car"),
        ({"bicycle","bike","cycle"}, "bicycle"),
        ({"bus","train","metro"}, "bus"),
        ({"person","man","woman","human","people","individual","adult","child"}, "person"),
        ({"keys","key","keychain","key ring"}, "keys"),
        ({"coin","coins","currency","money","cash","banknote","bill"}, "currency"),
        ({"towel","hand towel","bath towel","napkin"}, "towel"),
        ({"pillow","cushion","throw pillow"}, "pillow"),
        ({"soap","hand soap","bar soap","body wash"}, "soap"),
        ({"toothbrush"}, "toothbrush"),
        ({"toothpaste"}, "toothpaste"),
        ({"comb","hairbrush","hair brush"}, "comb"),
        ({"charger","cable","wire","cord","usb cable","power cable","adapter"}, "cable"),
        ({"broom","brush","sweeper"}, "broom"),
        ({"bucket","pail"}, "bucket"),
        ({"mop"}, "mop"),
        ({"candle"}, "candle"),
        ({"lighter","matchbox","matches"}, "lighter"),
        ({"ball","football","cricket ball","tennis ball"}, "ball"),
        ({"ring","necklace","bracelet","bangle","jewelry","jewellery","earring"}, "jewelry"),
        ({"plant","houseplant","flower pot","potted plant","flower"}, "plant"),
        ({"basket","laundry basket"}, "basket"),
        ({"lock","padlock"}, "lock"),
        ({"vase"}, "vase"),
        ({"pillowcase","bedsheet","bed sheet"}, "bedsheet"),
    ]

    def get_canonical(label):
        l = label.lower().strip()
        for group, canonical in SYNONYM_GROUPS:
            if l in group:
                return canonical
        # fallback: label contains a known synonym as a whole word/phrase
        # (e.g. "portable laptop" or "gaming laptop" -> laptop) so variants
        # Imagga returns don't slip through as separate unmapped duplicates
        for group, canonical in SYNONYM_GROUPS:
            for member in group:
                if re.search(r'\b' + re.escape(member) + r'\b', l):
                    return canonical
        # fallback: strip simple trailing plural and recheck, so unmapped
        # plural/singular pairs (e.g. "toy"/"toys") collapse to one entry
        # instead of showing as two different objects
        if l.endswith("s") and len(l) > 3 and not l.endswith("ss"):
            singular = l[:-1]
            for group, canonical in SYNONYM_GROUPS:
                if singular in group:
                    return canonical
            return singular
        return l

    seen_canonical = set()
    deduped = []
    for t in filtered:
        label = t["tag"]["en"].lower()
        if label in USELESS_TAGS:
            continue
        canonical = get_canonical(label)
        if canonical not in seen_canonical:
            seen_canonical.add(canonical)
            deduped.append(canonical)
        if len(deduped) == 5:
            break

    if not deduped:
        return jsonify({"result": "Could not identify any objects clearly."})

    top_tag_conf = filtered[0].get("confidence", 0) if filtered else 0
    if top_tag_conf < 55:
        result = "This might be: " + ", ".join(deduped) + ". Move closer or improve lighting for a clearer read."
    else:
        result = "I can see: " + ", ".join(deduped) + "."

    raw_debug = ", ".join(f"{t['tag']['en']}:{round(t.get('confidence',0),1)}" for t in tags[:5])
    log_activity(user_id, "object_scan", f"{result} | raw_top5: {raw_debug}")

    return jsonify({"result": result, "tags": deduped})

# ── Register — multi-angle (accepts images[] array OR single image) ────────────
@app.route("/register", methods=["POST"])
def register():
    data        = request.get_json()
    user_id     = data.get("user_id")
    name        = data.get("name", "").strip()
    description = data.get("description", "").strip()

    images_raw = data.get("images") or ([data.get("image")] if data.get("image") else [])
    if not all([user_id, name]) or not images_raw:
        return jsonify({"error": "Missing data"}), 400

    faceset_token = ensure_faceset(user_id, name)
    collected_tokens = []
    failed = 0

    for img_b64 in images_raw:
        if not img_b64:
            continue
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(img_b64)
            image_b64c  = compress_image_b64(image_bytes)
            detect_res  = facepp_post("/facepp/v3/detect", image_b64=image_b64c)
            if detect_res.get("error_message"):
                failed += 1
                continue
            faces = detect_res.get("faces", [])
            if not faces:
                failed += 1
                continue
            token = faces[0]["face_token"]
            facepp_post("/facepp/v3/faceset/addface", {
                "faceset_token": faceset_token,
                "face_tokens": token
            })
            collected_tokens.append(token)
        except Exception:
            failed += 1
            continue

    if not collected_tokens:
        return jsonify({"error": "No face detected in any of the provided images"}), 400

    tokens_str = ",".join(collected_tokens)
    db = get_db()
    existing = db.execute(
        "SELECT id, face_token FROM persons WHERE user_id=? AND name=?", (user_id, name)
    ).fetchone()
    if existing:
        prev = existing["face_token"] or ""
        merged = ",".join(dict.fromkeys((prev + "," + tokens_str).strip(",").split(",")))
        db.execute(
            "UPDATE persons SET face_token=?, description=? WHERE id=?",
            (merged, description, existing["id"])
        )
    else:
        db.execute(
            "INSERT INTO persons (user_id, name, face_token, description) VALUES (?,?,?,?)",
            (user_id, name, tokens_str, description)
        )
    db.commit()
    db.close()

    angle_word = f"{len(collected_tokens)} angle{'s' if len(collected_tokens) > 1 else ''}"
    log_activity(user_id, "register", f"{name} ({angle_word})")
    return jsonify({
        "result": f"{name} registered successfully with {angle_word}.",
        "tokens_saved": len(collected_tokens),
        "failed_images": failed
    })

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
        all_tokens = [t.strip() for t in face_token.split(",") if t.strip()]
        for tok in all_tokens:
            facepp_post("/facepp/v3/faceset/removeface", {
                "faceset_token": faceset_row["faceset_token"],
                "face_tokens": tok
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

@app.route("/users", methods=["GET"])
def list_users():
    db = get_db()
    users = db.execute("SELECT id, name FROM users ORDER BY name ASC").fetchall()
    db.close()
    return jsonify({"users": [{"id": r["id"], "name": r["name"]} for r in users]})

@app.route("/caregiver-dashboard")
def caregiver_dashboard():
    return send_file("caregiver_dashboard.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
