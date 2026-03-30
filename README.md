# 🐦 Dove-Detector

**Two-stage AI pigeon detector for your balcony.**

YOLO26 detects birds in ~50ms, CLIP classifies pigeon vs. other birds in ~80ms – fully CPU-based, no GPU needed, no fine-tuning required.

Sparrows, blackbirds and tits stay. Pigeons get the alarm. 🔊

---

## How it works

```
Camera frame (every ~1s)
       │
       ▼
 YOLO26m/l ──── no bird? ────→ sleep 🌙
       │
   bird detected
       │
       ▼
  CLIP ViT-B/32
  "pigeon or not?"
       │
  ┌────┴────┐
  │         │
PIGEON    other bird
  │         │
🔊 ALARM  ✅ ignore
```

**Stage 1 – YOLO26:** Detects any bird in the frame. Fast and lightweight, filters out 95% of frames instantly.

**Stage 2 – CLIP:** Only called when YOLO sees a bird. Classifies the cropped bird image: pigeon/dove or something else? Zero-shot, no training needed.

**Optional Stage 2b – Vision LLM:** Set `USE_CLIP = False` to use a Vision LLM via LM Studio (e.g. Qwen3-VL-4B) instead of CLIP. Much slower on CPU but more powerful.

---

## Requirements

- Python 3.11+
- CPU with ~2GB free RAM (no GPU needed)
- A camera – webcam or Android phone via [IP Webcam app](https://play.google.com/store/apps/details?id=com.pas.webcam)
- Optional: [LM Studio](https://lmstudio.ai) with a Vision LLM loaded (only if `USE_CLIP = False`)

---

## Installation

```bash
git clone https://github.com/lh-tech-ai/dove-detector
cd dove-detector
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install "numpy<2" torch torchvision ultralytics opencv-python requests pillow open-clip-torch
```

On first run, models are downloaded automatically:

- YOLO26l – ~84MB
- CLIP ViT-B/32 – ~605MB

---

## Configuration

Edit the config section at the top of `app.py`:

```python
# Camera
CAMERA_SNAPSHOT_URL = "http://192.168.1.XX:8080/shot.jpg"  # IP Webcam App URL - or - Camera Index (also configure camera index variable)
USE_SNAPSHOT = True                                          # recommended!

# Detection
YOLO_MODEL      = "yolo26m.pt"   # yolo26n / s / m / l
YOLO_CONFIDENCE = 0.20           # lower = more sensitive
CLIP_THRESHOLD  = 0.45           # higher = stricter pigeon detection

# Mode
USE_CLIP = True                  # True = CLIP (fast), False = Vision LLM (slow)

# Alarm
ALARM_SOUND          = "alarm.wav"   # any WAV file, e.g. a raptor scream
COOLDOWN_AFTER_ALARM = 30            # seconds before next alarm can trigger
```
**Pro tip:** The config in the .py file is already very good - better let it like this.

---

## Usage

```bash
python app.py
```

Example output for this image:

<img width="1280" height="853" alt="image" src="https://github.com/user-attachments/assets/f2ec7086-6b03-4d0d-bb89-6ee710e2f715" />

Output:
```
[13:04:40] No bird. (Checks: 2 | Doves: 0)                            
[13:04:41] 🐤 1 bird(s) recognized! → Checking with CLIP...
   Bird #1 (YOLO: 73%) → CLIP... 🕊️  DOVE DETECTED! (Dove, HIGH, 97% confidence) [Overall dove count: 1]
   💾 Saved: detections/20260330_130441_*.jpg
   🔊 ALERT played!
   ⏸️  Cooldown 30s...
```

Detected pigeons are saved as full frame + cropped bird image in `detections/`.

---

## Alarm Sound

Any WAV file works. A raptor/hawk scream is most effective against pigeons.

Free sounds: [freesound.org](https://freesound.org) → search "hawk scream" or "falcon scream".

```bash
# Quick test beep (Linux):
sudo apt install sox
sox -n alarm.wav synth 0.5 triangle 800
```

---

## Optional: Vision LLM via LM Studio

If you want to use a Vision LLM instead of CLIP, set `USE_CLIP = False` and load a vision model in LM Studio (recommended: `Qwen3-VL-4B-Instruct`, Q4_K_M, ~3.3GB).

Make sure the LM Studio server is running on `localhost:1234` and adjust `VISION_MODEL` to match the exact model name shown in LM Studio.

---

## Hardware tested on

| Device | Mode | Speed |
|---|---|---|
| Chromebook (Crostini/Linux) | CLIP, CPU-only | ~200ms/frame |
| Chromebook (Crostini/Linux) | Qwen3-VL-4B via LM Studio | ~12s/frame |

---

## Troubleshooting

**No camera found (`/dev/video*` missing):**
On Chromebook/Crostini, direct webcam access is blocked. Use the IP Webcam app on your Android phone instead – works even better since you can position it freely.

**NumPy version error:**
```bash
pip install "numpy<2"
```

**YOLO detects birds that aren't birds:**
Lower `YOLO_CONFIDENCE` is more sensitive but produces more false positives. CLIP will catch most of them in stage 2. `0.20` is a good starting point.

**CLIP misses pigeons:**
Lower `CLIP_THRESHOLD` (e.g. `0.35`) or add more pigeon label variants to `CLIP_LABELS`.

---

## License

MIT – do whatever you want with it. Just don't use it against sparrows. 🐦

*Built on a Chromebook. With a phone as a camera. Tested by pointing the phone at a picture of a pigeon on a monitor. AI is wild.*
