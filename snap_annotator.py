"""
╔══════════════════════════════════════════════╗
║   SnapAnnotator  —  Day 12 (Moderate)        ║
║   Local vision AI  •  offline  •  real-time  ║
╚══════════════════════════════════════════════╝

Setup:
    pip install opencv-python pillow ollama rich
    ollama pull moondream

Controls:
    S  —  Snapshot + analyze current frame
    C  —  Clear overlay
    Q  —  Quit

Mouse:
    Click any labeled box on the annotated window
    to ask a follow-up question about that object.
"""

import cv2
import ollama
import base64
import json
import time
import threading
import queue
import io
import re
import sys
from datetime import datetime
from pathlib import Path
from PIL import Image

# ─── try rich; fall back gracefully if not installed ─────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box as rbox
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class Console:
        def print(self, *a, **kw): print(*a)
        def rule(self, t=""): print("─" * 50 + f" {t}")
    console = Console()

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL          = "moondream"
CAPTURE_SIZE   = (512, 512)     # resize before inference — big impact on speed
DISPLAY_WIDTH  = 800            # live feed display width
SAVE_SNAPSHOTS = True           # save annotated frames to ./snapshots/
SESSION_LOG    = "session.json" # append all analyses here
FOLLOWUP_MODEL = "moondream"    # can swap to "llava" or "llama3.2-vision"
# ─────────────────────────────────────────────────────────────────────────────

SNAPSHOT_DIR = Path("snapshots")
if SAVE_SNAPSHOTS:
    SNAPSHOT_DIR.mkdir(exist_ok=True)

# Color palette for object boxes — one per object, cycles if >8 objects
COLORS = [
    (0, 255, 100),   # green
    (0, 180, 255),   # orange
    (255, 100, 0),   # blue
    (200, 0, 255),   # purple
    (0, 255, 255),   # yellow
    (255, 0, 150),   # pink
    (100, 255, 0),   # lime
    (255, 200, 0),   # cyan
]

ANALYSIS_PROMPT = """Analyze this image carefully.
Return ONLY valid JSON — no markdown, no explanation, nothing else.

{
  "description": "One clear sentence describing the whole scene",
  "mood": "single word: busy / calm / dark / bright / empty / crowded / natural / urban",
  "objects": [
    {
      "name": "object name",
      "position": "left / center / right / top-left / top-right / bottom-left / bottom-right / top / bottom",
      "details": "one useful detail about this object",
      "notable": true or false
    }
  ]
}

Be accurate. List every distinct visible object. Mark the most interesting one as notable: true."""


def strip_json(raw: str) -> str:
    """Moondream often wraps output in ```json ... ```. Strip it."""
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def parse_position(pos: str, frame_w: int, frame_h: int):
    """
    Convert a text position ('top-left', 'center', etc.)
    to pixel (cx, cy) coordinates.
    
    moondream doesn't return bounding boxes — this is a known limitation of
    small VLMs. We map text positions to approximate screen regions.
    This is good enough for a demo; llava or llama3.2-vision give better coords.
    """
    pos = pos.lower()
    col = (frame_w // 4  if "left"   in pos else
           3 * frame_w // 4 if "right" in pos else
           frame_w // 2)
    row = (frame_h // 4  if "top"    in pos else
           3 * frame_h // 4 if "bottom" in pos else
           frame_h // 2)
    return col, row


def encode_frame(frame) -> str:
    """BGR OpenCV frame → base64 JPEG string."""
    resized  = cv2.resize(frame, CAPTURE_SIZE)
    pil_img  = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    buf      = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def draw_hud(frame, last_result: dict | None, analyzing: bool):
    """
    Draw a persistent semi-transparent HUD on the live feed.
    Shows last description + object count, or a spinner while analyzing.
    """
    overlay = frame.copy()
    h, w    = frame.shape[:2]
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if analyzing:
        msg = "Analyzing..."
        cv2.putText(frame, msg, (12, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    elif last_result:
        desc = last_result.get("description", "")[:72]
        n    = len(last_result.get("objects", []))
        cv2.putText(frame, f"{desc}",    (12, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 255, 200), 1)
        cv2.putText(frame, f"{n} objects detected  |  S=new snapshot  Q=quit",
                    (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
    else:
        cv2.putText(frame, "S = snapshot & analyze   Q = quit",
                    (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)
    return frame


def draw_annotations(frame, result: dict) -> tuple:
    """
    Draw labeled boxes for each detected object.
    Returns (annotated_frame, list_of_hit_boxes).

    hit_boxes: list of (x1, y1, x2, y2, obj_dict) — used for click detection.
    """
    annotated = frame.copy()
    h, w      = frame.shape[:2]
    hit_boxes = []

    for i, obj in enumerate(result.get("objects", [])):
        color = COLORS[i % len(COLORS)]
        cx, cy = parse_position(obj.get("position", "center"), w, h)

        x1, y1 = cx - 70, cy - 45
        x2, y2 = cx + 70, cy + 45

        # semi-transparent fill
        sub = annotated[max(0,y1):y2, max(0,x1):x2]
        if sub.size > 0:
            filled = sub.copy()
            filled[:] = (int(color[0]*0.2), int(color[1]*0.2), int(color[2]*0.2))
            cv2.addWeighted(filled, 0.35, sub, 0.65, 0, sub)
            annotated[max(0,y1):y2, max(0,x1):x2] = sub

        # border — thicker if notable
        thickness = 3 if obj.get("notable") else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # label pill above box
        label    = obj.get("name", "?")[:18]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lx, ly   = x1, y1 - 6
        cv2.rectangle(annotated, (lx, ly - th - 6), (lx + tw + 10, ly + 2), color, -1)
        cv2.putText(annotated, label, (lx + 5, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # index badge
        cv2.circle(annotated, (x2 - 12, y1 + 12), 10, color, -1)
        cv2.putText(annotated, str(i + 1), (x2 - 16, y1 + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        hit_boxes.append((x1, y1, x2, y2, obj))

    # scene description banner at top
    desc = result.get("description", "")[:70]
    mood = result.get("mood", "")
    cv2.rectangle(annotated, (0, 0), (w, 40), (20, 20, 20), -1)
    cv2.putText(annotated, f"{desc}  [{mood}]", (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    return annotated, hit_boxes


def print_result(result: dict, latency: float):
    """Pretty-print the analysis to terminal."""
    if HAS_RICH:
        console.rule("[bold green]SNAP ANALYSIS[/bold green]")
        console.print(Panel(
            f"[italic]{result.get('description', '')}[/italic]\n"
            f"Mood: [bold]{result.get('mood', '')}[/bold]",
            title="Scene", border_style="green"
        ))
        t = Table(box=rbox.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
        t.add_column("#",        width=3)
        t.add_column("Object",   width=18)
        t.add_column("Position", width=14)
        t.add_column("Details")
        for i, obj in enumerate(result.get("objects", [])):
            star = "★" if obj.get("notable") else ""
            t.add_row(str(i+1), f"{star}{obj.get('name','?')}",
                      obj.get("position", ""), obj.get("details", ""))
        console.print(t)
        console.print(f"[dim]⏱  {latency:.1f}s   •   "
                      f"Click a box in the annotated window to ask a follow-up[/dim]\n")
    else:
        print("\n" + "="*55)
        print("SCENE:", result.get("description",""))
        print("MOOD :", result.get("mood",""))
        print("-"*55)
        for i, obj in enumerate(result.get("objects",[])):
            star = "★" if obj.get("notable") else " "
            print(f" {star}{i+1}. {obj.get('name','?'):18} "
                  f"[{obj.get('position','')}]  {obj.get('details','')}")
        print(f"\n⏱  {latency:.1f}s   Click a box to ask a follow-up.\n")


def run_analysis(frame, result_holder: dict, q: queue.Queue):
    """
    Background thread: encode frame, call moondream, parse result.
    Puts result into result_holder and signals q when done.
    """
    try:
        img_b64 = encode_frame(frame)
        t0      = time.time()
        resp    = ollama.chat(
            model=MODEL,
            messages=[{
                "role":    "user",
                "content": ANALYSIS_PROMPT,
                "images":  [img_b64]
            }]
        )
        latency = time.time() - t0
        raw     = strip_json(resp["message"]["content"])
        result  = json.loads(raw)
        result_holder["data"]    = result
        result_holder["frame"]   = frame
        result_holder["img_b64"] = img_b64
        result_holder["latency"] = latency
        q.put(("ok", result, latency))
    except json.JSONDecodeError as e:
        raw_text = resp["message"]["content"] if "resp" in dir() else "no response"
        q.put(("json_err", raw_text, 0))
    except Exception as e:
        q.put(("err", str(e), 0))


def run_followup(question: str, img_b64: str, description: str, q: queue.Queue):
    """Background thread for follow-up questions."""
    try:
        resp = ollama.chat(
            model=FOLLOWUP_MODEL,
            messages=[{
                "role":    "user",
                "content": f"{question}\n\nContext: {description}",
                "images":  [img_b64]
            }]
        )
        q.put(("followup_ok", resp["message"]["content"]))
    except Exception as e:
        q.put(("followup_err", str(e)))


def save_snapshot(frame, result: dict):
    """Save annotated frame + JSON sidecar to ./snapshots/."""
    ts   = datetime.now().strftime("%H%M%S")
    stem = SNAPSHOT_DIR / f"snap_{ts}"
    cv2.imwrite(str(stem) + ".jpg", frame)
    with open(str(stem) + ".json", "w") as f:
        json.dump(result, f, indent=2)
    return str(stem) + ".jpg"


def append_session_log(result: dict):
    log = []
    if Path(SESSION_LOG).exists():
        with open(SESSION_LOG) as f:
            try: log = json.load(f)
            except: log = []
    log.append({"time": datetime.now().isoformat(), **result})
    with open(SESSION_LOG, "w") as f:
        json.dump(log, f, indent=2)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam. Close other camera apps and try again.")
        sys.exit(1)

    # scale display
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # shared state
    last_result:  dict        = {}   # {"data", "frame", "img_b64", "latency"}
    annotated_frame           = None
    hit_boxes:    list        = []
    analyzing:    bool        = False
    worker_q:     queue.Queue = queue.Queue()
    followup_q:   queue.Queue = queue.Queue()
    active_obj:   dict | None = None  # object waiting for a question

    # ── mouse callback ────────────────────────────────────────────────────────
    def on_mouse(event, x, y, flags, param):
        nonlocal active_obj
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for (x1, y1, x2, y2, obj) in hit_boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                active_obj = obj
                if HAS_RICH:
                    console.print(
                        f"\n[bold yellow]Clicked:[/bold yellow] "
                        f"[cyan]{obj['name']}[/cyan] — type your question in the terminal:"
                    )
                else:
                    print(f"\nClicked: {obj['name']} — type your question:")
                break

    cv2.namedWindow("SnapAnnotator — Live Feed")
    cv2.namedWindow("Annotated Snapshot")
    cv2.setMouseCallback("Annotated Snapshot", on_mouse)

    if HAS_RICH:
        console.print(Panel(
            "[bold]S[/bold] — snapshot & analyze\n"
            "[bold]C[/bold] — clear overlay\n"
            "[bold]Q[/bold] — quit\n\n"
            "[dim]Click any labeled box → type a follow-up question[/dim]",
            title="[bold green]SnapAnnotator[/bold green]",
            border_style="green"
        ))
    else:
        print("\n🚀 SnapAnnotator  |  S=snapshot  C=clear  Q=quit")
        print("   Click any labeled box in the snapshot window to ask a follow-up.\n")

    # ── input thread (non-blocking follow-up questions) ───────────────────────
    input_q: queue.Queue = queue.Queue()

    def input_listener():
        while True:
            try:
                line = input()
                input_q.put(line)
            except EOFError:
                break

    threading.Thread(target=input_listener, daemon=True).start()

    # ── main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # resize for display
        dh = int(frame.shape[0] * DISPLAY_WIDTH / frame.shape[1])
        display = cv2.resize(frame, (DISPLAY_WIDTH, dh))
        display = draw_hud(display, last_result.get("data"), analyzing)
        cv2.imshow("SnapAnnotator — Live Feed", display)

        # ── check worker results ──────────────────────────────────────────────
        try:
            msg = worker_q.get_nowait()
            analyzing = False
            status    = msg[0]
            if status == "ok":
                _, result, latency = msg
                last_result["data"]    = result
                last_result["img_b64"] = last_result.get("img_b64", "")
                print_result(result, latency)

                ann, hit_boxes = draw_annotations(last_result["frame"], result)
                annotated_frame = ann
                cv2.imshow("Annotated Snapshot", ann)

                if SAVE_SNAPSHOTS:
                    path = save_snapshot(ann, result)
                    if HAS_RICH:
                        console.print(f"[dim]💾 Saved → {path}[/dim]")
                    else:
                        print(f"💾 Saved → {path}")
                append_session_log(result)

            elif status == "json_err":
                raw = msg[1]
                if HAS_RICH:
                    console.print("[red]⚠  JSON parse failed. Raw moondream output:[/red]")
                    console.print(Panel(raw[:600], border_style="red"))
                else:
                    print("⚠  JSON parse failed:")
                    print(raw[:400])

            elif status == "err":
                print(f"❌ Error: {msg[1]}")
        except queue.Empty:
            pass

        # ── check follow-up results ───────────────────────────────────────────
        try:
            fmsg = followup_q.get_nowait()
            if fmsg[0] == "followup_ok":
                if HAS_RICH:
                    console.print(Panel(
                        fmsg[1],
                        title=f"[bold cyan]Answer about {active_obj['name']}[/bold cyan]",
                        border_style="cyan"
                    ))
                else:
                    print(f"\n💬 Answer: {fmsg[1]}\n")
            else:
                print(f"❌ Follow-up error: {fmsg[1]}")
            active_obj = None
        except queue.Empty:
            pass

        # ── check for typed follow-up ─────────────────────────────────────────
        try:
            question = input_q.get_nowait()
            if active_obj and question.strip() and last_result.get("img_b64"):
                if HAS_RICH:
                    console.print(f"[dim]Asking moondream about "
                                  f"{active_obj['name']}...[/dim]")
                else:
                    print(f"Asking about {active_obj['name']}...")
                threading.Thread(
                    target=run_followup,
                    args=(question.strip(),
                          last_result["img_b64"],
                          last_result["data"].get("description", ""),
                          followup_q),
                    daemon=True
                ).start()
        except queue.Empty:
            pass

        # ── key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c'):
            hit_boxes = []
            annotated_frame = None
            last_result.clear()
            cv2.destroyWindow("Annotated Snapshot")

        elif key == ord('s') and not analyzing:
            analyzing = True
            snap = frame.copy()
            last_result["frame"]   = snap
            last_result["img_b64"] = encode_frame(snap)  # cache now for follow-ups

            if HAS_RICH:
                console.print(f"[bold green]📸 Analyzing frame with {MODEL}...[/bold green]")
            else:
                print(f"\n📸 Analyzing with {MODEL}...")

            threading.Thread(
                target=run_analysis,
                args=(snap, last_result, worker_q),
                daemon=True
            ).start()

    cap.release()
    cv2.destroyAllWindows()
    if HAS_RICH:
        console.print("\n[bold]👋 SnapAnnotator closed.[/bold]")
    else:
        print("\n👋 SnapAnnotator closed.")


if __name__ == "__main__":
    main()
