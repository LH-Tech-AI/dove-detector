"""
🐦 Dove-Detector v2 – Hybrid: YOLOv8n (bird?) → CLIP (pigeon/dove?)
For CPU computer with external camera (e.g. smartphone or webcam). No GPU needed. Installs ~615MB of AI model data on first run.
"""

print("Loading 🐦 Dove-Detector v2 – Hybrid YOLO + CLIP ...")

import time
import base64
import io
import sys
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image
from ultralytics import YOLO

import open_clip
import torch

# ─── Config ───────────────────────────────────────────────

CAMERA_INDEX = "http://192.168.178.59:8080/video"
CAMERA_SNAPSHOT_URL = "http://192.168.178.59:8080/shot.jpg"
USE_SNAPSHOT = True  # True = shot.jpg (MUCH better!), False = MJPEG stream (very bad!)
CHECK_INTERVAL_NO_BIRD = 1
CHECK_INTERVAL_BIRD = 0.5
COOLDOWN_AFTER_ALARM = 30

YOLO_MODEL = "yolo26l.pt"
YOLO_CONFIDENCE = 0.20
YOLO_BIRD_CLASS_ID = 14

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
VISION_MODEL = "qwen/qwen3-vl-4b"
CROP_PADDING = 40
CROP_MAX_SIZE = 384

USE_CLIP = True  # True = CLIP (fast), False = Vision LLM (very slow on small devices without GPU!)

ALARM_SOUND = "alarm.wav"
SAVE_DETECTIONS = True
SAVE_DIR = Path("detections")

# ─── CLIP ─────────────────────────────────────────────────────────

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

CLIP_LABELS = [
    "a pigeon",
    "a dove",
    "a feral pigeon on a balcony",
    "a rock dove",
    "a sparrow",
    "a blackbird",
    "a tit bird",
    "a robin",
    "a finch",
    "a crow",
    "a squirrel",
    "an unknown bird",
]
CLIP_PIGEON_LABELS = {"a pigeon", "a dove", "a feral pigeon on a balcony", "a rock dove"}
CLIP_THRESHOLD = 0.45


def init_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model.eval()

    text_tokens = tokenizer(CLIP_LABELS)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return model, preprocess, text_features


def ask_clip(model, preprocess, text_features, image: np.ndarray) -> dict:
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        img_features = model.encode_image(img_tensor)
        img_features /= img_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * img_features @ text_features.T).softmax(dim=-1)[0]

    best_idx = probs.argmax().item()
    best_label = CLIP_LABELS[best_idx]
    best_prob = probs[best_idx].item()

    pigeon_prob = sum(
        probs[i].item() for i, l in enumerate(CLIP_LABELS) if l in CLIP_PIGEON_LABELS
    )

    is_pigeon = pigeon_prob >= CLIP_THRESHOLD

    if pigeon_prob >= 0.7:
        confidence = "HIGH"
    elif pigeon_prob >= 0.45:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "is_pigeon": is_pigeon,
        "species": best_label.replace("a ", "").title(),
        "confidence": confidence,
        "pigeon_prob": f"{pigeon_prob:.0%}",
        "raw": f"{best_label} ({best_prob:.0%}), pigeon_total={pigeon_prob:.0%}",
    }

# ─── Sound ────────────────────────────────────────────────────────

def play_alarm():
    if not Path(ALARM_SOUND).exists():
        print("   🔊 ALERT! (no soundfile found - only text alert)")
        return

    system = platform.system()
    try:
        if system == "Linux":
            subprocess.Popen(
                ["aplay", "-q", ALARM_SOUND],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif system == "Darwin":
            subprocess.Popen(["afplay", ALARM_SOUND])
        elif system == "Windows":
            subprocess.Popen(
                ["powershell", "-c",
                 f"(New-Object Media.SoundPlayer '{ALARM_SOUND}').PlaySync()"]
            )
        print("   🔊 ALERT played!")
    except Exception as e:
        print(f"   🔊 Sound-Error: {e}")

# ─── Camera ───────────────────────────────────────────────────────

def init_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"❌ Camera {index} failed to open!")
        print("   Tip: use other camera index or setup the IP Webcam App on your smartphone.")
        sys.exit(1)
    # Auflösung etwas runtersetzen für Performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📷 Camera {index} opened ({w}x{h})")
    return cap


def grab_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    ret, frame = cap.read()
    return frame if ret else None

# ─── YOLO ─────────────────────────────────────────────────────────

def detect_birds(model: YOLO, frame: np.ndarray) -> list:
    results = model(frame, verbose=False, conf=YOLO_CONFIDENCE)
    birds = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == YOLO_BIRD_CLASS_ID:
                birds.append(box)
    return birds


def crop_bird(frame: np.ndarray, box, padding: int = CROP_PADDING) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return frame[y1:y2, x1:x2]

# ─── Vision LLM ──────────────────────────────────────────────────

def image_to_base64(image: np.ndarray, max_size: int = CROP_MAX_SIZE) -> str:
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


PIGEON_PROMPT = """You see a cropped image of a bird. Identify it.
Is this bird a PIGEON or DOVE (German: Taube)?
Common pigeons include: rock dove, feral pigeon, wood pigeon, collared dove, stock dove.
NOT a pigeon/dove: sparrow, blackbird, tit, robin, finch, starling, crow, magpie, etc.

Answer in EXACTLY this format (nothing else):
BIRD: <species>
PIGEON: <YES or NO>
SURE: <HIGH, MEDIUM, or LOW>"""


def ask_vision_llm(image_b64: str) -> dict:
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PIGEON_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 60,
        "temperature": 0.1,
    }

    try:
        resp = requests.post(LM_STUDIO_URL, json=payload, timeout=120)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return parse_response(text)
    except requests.exceptions.ConnectionError:
        return {"error": "LM Studio not reachable! Server started?",
                "is_pigeon": False, "raw": ""}
    except requests.exceptions.Timeout:
        return {"error": "LM Studio Timeout (>120s)", "is_pigeon": False, "raw": ""}
    except Exception as e:
        return {"error": str(e), "is_pigeon": False, "raw": ""}


def parse_response(text: str) -> dict:
    result = {
        "raw": text,
        "is_pigeon": False,
        "species": "unknown",
        "confidence": "LOW",
    }
    for line in text.upper().splitlines():
        line = line.strip()
        if line.startswith("BIRD:"):
            result["species"] = line.split("BIRD:")[-1].strip().title()
        elif line.startswith("PIGEON:"):
            result["is_pigeon"] = "YES" in line
        elif line.startswith("SURE:"):
            for level in ("HIGH", "MEDIUM", "LOW"):
                if level in line:
                    result["confidence"] = level
                    break
    return result

# ─── Save ────────────────────────────────────────────────────

def save_detection(frame: np.ndarray, crop: np.ndarray, result: dict):
    SAVE_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(str(SAVE_DIR / f"{ts}_full.jpg"), frame)
    cv2.imwrite(str(SAVE_DIR / f"{ts}_crop.jpg"), crop)
    info = f"{ts} | {result.get('species', '?')} | {result.get('confidence', '?')}"
    with open(SAVE_DIR / "log.txt", "a") as f:
        f.write(info + "\n")
    print(f"   💾 Saved: {SAVE_DIR}/{ts}_*.jpg")

def grab_frame_http(url: str) -> np.ndarray | None:
    try:
        resp = requests.get(url, timeout=5)
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

# ─── Main loop ───────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  🐦 Dove-Detector v2 – Hybrid YOLO + CLIP")
    print("=" * 55)
    print(f"  YOLO:       {YOLO_MODEL}")
    print(f"  CLIP:       {CLIP_MODEL_NAME}")
    print(f"  Use CLIP:   {'Yes' if USE_CLIP else 'No'}")
    print(f"  Vision LLM: {VISION_MODEL}")
    print(f"  LM Studio:  {LM_STUDIO_URL}")
    print(f"  Intervall:  {CHECK_INTERVAL_NO_BIRD}s (idle) / "
          f"{CHECK_INTERVAL_BIRD}s (bird)")
    print(f"  Cooldown:   {COOLDOWN_AFTER_ALARM}s after alert")
    print(f"  Save:       {'Yes' if SAVE_DETECTIONS else 'No'}")
    print("-" * 55)

    # Load YOLO (auto-download on first run, then cached)
    print("⏳ Loading YOLO-Modell...")
    model = YOLO(YOLO_MODEL)
    print("✅ YOLO loaded.\n")

    # Load CLIP (if activated)
    clip_model = clip_preprocess = clip_text_features = None
    if USE_CLIP:
        print("⏳ Loading CLIP-Model...")
        clip_model, clip_preprocess, clip_text_features = init_clip()
        print("✅ CLIP loaded\n")
    else:
        print("⏳ Testing LM Studio connection...")
        try:
            r = requests.get(
                LM_STUDIO_URL.replace("/chat/completions", "/models"),
                timeout=5,
            )
            r.raise_for_status()
            models = [m["id"] for m in r.json().get("data", [])]
            print(f"✅ LM Studio connected. Models: {models}")
            if models and VISION_MODEL not in models:
                print(f"⚠️  '{VISION_MODEL}' not found!")
                print(f"   Available: {models}")
                print(f"   → Adjust VISION_MODEL in the config above!")
                ACTUAL_MODEL = models[0] if models else VISION_MODEL
                print(f"   → Use instead: '{ACTUAL_MODEL}'")
        except Exception as e:
            print(f"⚠️  LM Studio not reachable: {e}")
            print("   → You need to start LM Studio and load a vision model (e.g. Qwen 3 VL 4B)!")
    print()

    cap = None
    if not USE_SNAPSHOT:
        cap = init_camera(CAMERA_INDEX)
    else:
        print(f"📷 Snapshot-Mode: {CAMERA_SNAPSHOT_URL}")

    dove_count = 0
    total_checks = 0

    print("\n🟢 Detector active. Ctrl+C to stop.\n")

    try:
        while True:
            if USE_SNAPSHOT:
                frame = grab_frame_http(CAMERA_SNAPSHOT_URL)
            else:
                frame = grab_frame(cap)

            if frame is None:
                print("⚠️  No frame received, retrying... Is your camera on?")
                time.sleep(1)
                continue

            total_checks += 1
            ts = datetime.now().strftime("%H:%M:%S")

            birds = detect_birds(model, frame)

            if not birds:
                status = (f"[{ts}] No bird. "
                          f"(Checks: {total_checks} | "
                          f"Doves: {dove_count})")
                print(f"\r{status: <70}", end="", flush=True)
                time.sleep(CHECK_INTERVAL_NO_BIRD)
                continue

            print(f"\n[{ts}] 🐤 {len(birds)} bird(s) recognized! "
                  f"→ {'Checking with CLIP' if USE_CLIP else 'Asking Vision-LLM'}...")

            pigeon_found = False

            birds = sorted(birds, key=lambda b: float(b.conf[0]), reverse=True)

            for i, box in enumerate(birds):
                conf = float(box.conf[0])
                crop = crop_bird(frame, box)

                if crop.shape[0] < 30 or crop.shape[1] < 30:
                    print(f"   Bird #{i+1}: too small, skipped")
                    continue

                classifier = "CLIP" if USE_CLIP else "LLM"
                print(f"   Bird #{i+1} (YOLO: {conf:.0%}) → {classifier}... ",
                      end="", flush=True)

                if USE_CLIP:
                    result = ask_clip(clip_model, clip_preprocess,
                                      clip_text_features, crop)
                else:
                    img_b64 = image_to_base64(crop)
                    result = ask_vision_llm(img_b64)

                if "error" in result:
                    print(f"❌ {result['error']}")
                    continue

                if result["is_pigeon"]:
                    dove_count += 1
                    pigeon_found = True
                    prob_info = f", {result['pigeon_prob']}" if "pigeon_prob" in result else ""
                    print(f"🕊️  DOVE DETECTED! ({result['species']}, "
                          f"{result['confidence']}{prob_info} confidence) "
                          f"[Overall dove count: {dove_count}]")
                    if SAVE_DETECTIONS:
                        save_detection(frame, crop, result)
                    break

                else:
                    print(f"✅ No problem ({result['species']}, "
                          f"{result['confidence']} confidence)")

            if pigeon_found:
                play_alarm()
                print(f"   ⏸️  Cooldown {COOLDOWN_AFTER_ALARM}s...")
                time.sleep(COOLDOWN_AFTER_ALARM)
            else:
                time.sleep(CHECK_INTERVAL_BIRD)

    except KeyboardInterrupt:
        print(f"\n\n{'=' * 55}")
        print(f"  🛑 Stopped.")
        print(f"  Checks: {total_checks} | Doves recognized: {dove_count}")
        if SAVE_DETECTIONS and dove_count > 0:
            print(f"  Saved images: {SAVE_DIR.absolute()}/")
        print(f"{'=' * 55}")
    finally:
        if cap:
            cap.release()

if __name__ == "__main__":
    main()
