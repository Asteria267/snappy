# 📸 SnapAnnotator — Day 12

> Point your webcam at anything. Press SPACE. Watch a local AI tell you exactly what it sees — **no internet, no API key, no data leaving your machine.**

Built for the **BuildCored Orcas** challenge. This is Day 12. 🐋

---

## 🤔 What does it actually do?

You open your webcam. You press `SPACE`. SnapAnnotator:

1. **Grabs** the frame from your camera
2. **Resizes & encodes** it as a JPEG, converts it to base64
3. **Sends** it to [Moondream](https://github.com/vikhyat/moondream) — a tiny vision AI running *locally* via [Ollama](https://ollama.com)
4. **Gets back** a structured JSON response with a scene description + every object the AI can see, including *where* each object is in the frame
5. **Draws numbered circles** on the actual camera frame at each object's position
6. **Renders a live sidebar** with the description, clickable object tags, and follow-up answers

All of this happens without freezing your UI — because the AI runs in a **background thread** while your camera feed stays live. Press `1`–`9` or click any tag to ask a focused follow-up question about any specific object. Press `E` to export the annotated frame. Press `C` to reset. Simple.

---

## 🧠 The Hardware Concept (why this is cool)

Real embedded AI systems — think NVIDIA Jetson Nano, Raspberry Pi with a Coral TPU, factory floor inspection cameras — follow this exact pipeline:

```
SENSOR → ISP → MEMORY BUFFER → INFERENCE ENGINE → ACTION / OUTPUT
```

SnapAnnotator mirrors that pipeline on your laptop:

```
Webcam → OpenCV → NumPy array → Moondream (via Ollama) → Annotated HUD
```

The model and the hardware differ. The **shape of the pipeline is identical**. When you press SPACE, you're doing the same thing a $200 industrial vision module does — just with Python instead of C++ and a MacBook instead of a Jetson. That's the point. 🤓

---

## ✨ Features

- 📷 **Live webcam feed** that never freezes, even while the AI is thinking
- 🧠 **Moondream vision model** running 100% offline on your machine
- 🗺️ **Spatial awareness** — the AI tells us *where* each object is (`top-left`, `center`, `bottom-right`, etc.) and we draw numbered markers right on the frame at those positions
- 🏷️ **Clickable object tags** in the sidebar — click any tag or press `1`–`9` for a follow-up
- 💬 **Follow-up Q&A** — ask the AI for detail about any specific detected object
- 💾 **Export** — press `E` to save the annotated frame to `./exports/snap_HHMMSS.jpg`
- ⚡ **Non-blocking inference** via Python threads + a message queue (UI stays smooth)
- 🎨 **Custom-rendered HUD** — rounded tags, animated scan-line during analysis, pulsing LIVE dot, flash effect on capture, hover states on tags
- 🔁 **Fallback parser** — if the model goes rogue and returns plain text instead of JSON, a regex parser still pulls the object names out
- 🎛️ **`--model` flag** — swap in any Ollama vision model from the command line without touching the code

---

## 🛠️ Tech Stack

| Tool | What it does here |
|---|---|
| `opencv-python` | Webcam capture, frame rendering, all the drawing |
| `pillow` | JPEG encoding before base64 conversion |
| `ollama` | Python client to talk to the local Ollama server |
| `numpy` | Canvas compositing, alpha blending, array math |
| Ollama | Local inference runtime — runs Moondream on your machine |
| Moondream | The actual vision-language model (~829 MB, downloaded once) |

---

## ⚙️ How the code is structured

The whole thing is **one Python file** — no packages, no config files to hunt down. Here's what's inside:

```
snap_annotator.py
│
├── 🎨  Drawing helpers      — fill(), rrect(), txt(), wrap(), spinner()
│                              (a mini 2D rendering engine built on OpenCV)
│
├── 🗺️  Marker overlay       — pos_to_xy() maps "top-left" → pixel coords
│                              draw_object_markers() draws numbered circles
│
├── 🧠  LLM layer            — analyse_frame() sends image + prompt to Moondream
│                              followup_query() asks about a specific object
│                              _parse_objects_fallback() handles bad JSON
│
├── 🖼️  Sidebar renderer     — render_sidebar() draws the entire right panel
│                              including flow-layout tag chips and follow-up area
│
└── 🔄  Main loop            — camera read, thread workers, queue draining,
                               canvas composition, keypress handling
```

**Why one file?** Because this is a learning project. Everything is visible in one scroll. You can read it top to bottom and understand the whole system without chasing imports.

**Why threads + a queue instead of async?** OpenCV's `waitKey` loop is synchronous. Threading with `queue.Queue` lets us run inference in the background without rewriting the render loop around `asyncio` — which doesn't play nicely with `cv2.imshow` anyway.

---

## 📦 Installation

### Step 1 — Install Ollama

Grab it from [ollama.com](https://ollama.com). It's a one-click installer on Mac, a shell script on Linux, and an `.exe` on Windows.

### Step 2 — Pull Moondream

```bash
ollama pull moondream
```

~829 MB download, happens once. Go make a coffee. ☕

### Step 3 — Install Python dependencies

```bash
pip install opencv-python pillow ollama numpy
```

Or, if you want to keep things clean (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
pip install opencv-python pillow ollama numpy
```

---

## ▶️ Running it

**Terminal 1** — start Ollama:
```bash
ollama serve
```

**Terminal 2** — run the app:
```bash
python snap_annotator.py
```

Want to try a different model? Easy:
```bash
python snap_annotator.py --model llava
```

---

## ⌨️ Controls

| Key / Action | What happens |
|---|---|
| `SPACE` | Capture frame → send to AI → render annotations |
| `1` – `9` | Ask a follow-up about object number N |
| `E` | Export annotated frame to `./exports/` |
| `C` | Clear everything, go back to live view |
| Click a tag | Same as pressing the matching number key |
| `Q` or `ESC` | Quit |

---

## 🐛 Troubleshooting

**`can't reach ollama`** — Ollama isn't running. Go to Terminal 1 and run `ollama serve`, then retry.

**`moondream not found`** — You haven't pulled the model yet. Run `ollama pull moondream` and wait for the download.

**Webcam won't open** — Something else is using your camera (Zoom, FaceTime, etc.). Close it and retry. On Linux you might need to change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`.

**Analysis is really slow** — Normal on CPU. The image is already resized to 512px before being sent. You can drop `INF_SIZE = 512` to `384` at the top of the file for faster (but slightly less detailed) results.

**Model returns weird text instead of JSON** — Happens sometimes with smaller models. The fallback parser handles it — you'll still get object names, just without position info.

---

## 📁 Project structure

```
snappy/
├── snap_annotator.py   ← the whole app
├── exports/            ← created automatically when you first press E
└── README.md
```

---

## 🔒 Privacy note

Zero network calls outside `localhost`. Ollama runs entirely on your machine. Your webcam frames go nowhere except into your own RAM and into the local model. You could run this on a plane with no wifi and it would work fine.

---

*Day 12 / BuildCored Orcas 🐋 — onward to Day 13!*
