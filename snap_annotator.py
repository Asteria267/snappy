"""
================================================================================
  SnapAnnotator — Day 12  Enhanced Edition  🐋
================================================================================
  SPACE   — capture & analyse
  A       — toggle auto-capture (captures every 5 s automatically)
  1–9     — follow-up query about object N
  E       — export annotated frame  →  ./exports/
  C       — clear / return to live view
  Q / ESC — quit

  Mouse:
    Click anywhere on the CAMERA  → ask "what's at this spot?"
    Click a tag in the SIDEBAR    → follow-up query for that object

  Prerequisites:
      pip install opencv-python pillow ollama numpy
      ollama pull moondream
      ollama serve          ← separate terminal

  What's fixed / added vs the basic version:
    ✓ Non-blocking — AI runs in background threads, UI never freezes
    ✓ Click camera frame to ask about any location (not just sidebar tags)
    ✓ Auto-capture mode with live countdown ring (press A)
    ✓ History strip — last 4 captures shown as thumbnails at the bottom
    ✓ Thread-safe state with a Lock (no race conditions on double-tap)
    ✓ Bug fixed: raw_text initialised before try block (was a silent failure)
    ✓ Position map is a clean dict instead of nested ternaries
    ✓ Sidebar split into focused sub-functions — easy to read and extend
    ✓ --model and --auto-interval CLI flags
================================================================================
"""

import argparse, base64, io, json, math, queue
import re, sys, threading, time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import ollama


# ─── layout ───────────────────────────────────────────────────────────────────
CAM_W         = 1280
CAM_H         = 720
SIDEBAR_W     = 380
WIN_W         = CAM_W + SIDEBAR_W
WIN_H         = CAM_H
INF_SIZE      = 512           # max px on longest side before sending to model
EXPORT_DIR    = Path("exports")
AUTO_INTERVAL = 5.0           # default seconds between auto-captures
HISTORY_MAX   = 4             # thumbnails kept in history strip
THUMB_W       = 88
THUMB_H       = 56


# ─── palette (BGR) ────────────────────────────────────────────────────────────
P = {
    "bg0":     ( 10,  10,  18),
    "bg1":     ( 18,  20,  30),
    "bg2":     ( 26,  28,  42),
    "border":  ( 48,  52,  78),
    "accent":  (210, 190,   0),
    "green":   ( 50, 200,  80),
    "amber":   ( 30, 165, 230),
    "purple":  (200,  90, 180),
    "auto":    ( 60, 220, 120),
    "txt0":    (240, 240, 250),
    "txt1":    (155, 160, 185),
    "txt2":    ( 80,  86, 112),
    "tag_bg":  ( 30,  36,  56),
    "tag_bdr": ( 65,  75, 115),
    "flash":   (255, 255, 255),
}

MARKER_COLORS = [
    (  0, 220, 255), (  0, 200, 100), (220, 100,   0),
    (200,   0, 220), (  0, 180, 220), (100, 220,   0),
    (220, 180,   0), (  0, 120, 255),
]

# Maps the model's position string → (x_fraction, y_fraction) of the frame.
# A plain dict is cleaner and easier to extend than nested ternaries.
POSITION_MAP = {
    "top-left":     (0.18, 0.18),
    "top":          (0.50, 0.18),
    "top-right":    (0.82, 0.18),
    "left":         (0.18, 0.50),
    "center":       (0.50, 0.50),
    "right":        (0.82, 0.50),
    "bottom-left":  (0.18, 0.82),
    "bottom":       (0.50, 0.82),
    "bottom-right": (0.82, 0.82),
}

F  = cv2.FONT_HERSHEY_SIMPLEX
FB = cv2.FONT_HERSHEY_DUPLEX


# ─── thread safety ────────────────────────────────────────────────────────────
_lock = threading.Lock()

def locked(st: dict, **kwargs):
    """Thread-safe batch update of the shared state dict."""
    with _lock:
        st.update(kwargs)


# ─── drawing helpers ──────────────────────────────────────────────────────────

def fill(img, x, y, w, h, color, alpha=1.0):
    x, y, w, h = int(x), int(y), int(w), int(h)
    sub = img[y:y+h, x:x+w]
    if sub.shape[0] == 0 or sub.shape[1] == 0:
        return
    if alpha >= 1.0:
        img[y:y+h, x:x+w] = color
    else:
        cv2.addWeighted(np.full_like(sub, color), alpha, sub, 1.0-alpha, 0, sub)
        img[y:y+h, x:x+w] = sub


def _rr_solid(img, x, y, w, h, r, c):
    cv2.rectangle(img, (x+r, y),   (x+w-r, y+h),   c, -1)
    cv2.rectangle(img, (x,   y+r), (x+w,   y+h-r), c, -1)
    for cx, cy in [(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
        cv2.circle(img, (cx,cy), r, c, -1)


def _rr_border(img, x, y, w, h, r, c, t):
    cv2.line(img, (x+r, y),   (x+w-r, y),   c, t)
    cv2.line(img, (x+r, y+h), (x+w-r, y+h), c, t)
    cv2.line(img, (x,   y+r), (x,   y+h-r), c, t)
    cv2.line(img, (x+w, y+r), (x+w, y+h-r), c, t)
    for cx, cy, a in [(x+r,y+r,180),(x+w-r,y+r,270),(x+r,y+h-r,90),(x+w-r,y+h-r,0)]:
        cv2.ellipse(img, (cx,cy), (r,r), 0, a, a+90, c, t)


def rrect(img, x, y, w, h, r, color, alpha=1.0, bcolor=None, bthick=1):
    """Rounded filled rectangle with optional border."""
    x,y,w,h,r = int(x),int(y),int(w),int(h), min(int(r), w//2, h//2)
    if alpha < 1.0:
        ov = img.copy()
        _rr_solid(ov, x, y, w, h, r, color)
        cv2.addWeighted(ov, alpha, img, 1.0-alpha, 0, img)
    else:
        _rr_solid(img, x, y, w, h, r, color)
    if bcolor:
        _rr_border(img, x, y, w, h, r, bcolor, bthick)


def txt(img, text, x, y, color, scale=0.46, thick=1, font=F):
    cv2.putText(img, str(text), (int(x),int(y)), font, scale, color, thick, cv2.LINE_AA)


def txt_w(text, scale=0.46, thick=1):
    (w, _), _ = cv2.getTextSize(str(text), F, scale, thick)
    return w


def wrap(text, maxc, maxl=99):
    lines, cur = [], ""
    for word in str(text).split():
        cand = (cur + " " + word).strip()
        if len(cand) > maxc:
            if cur: lines.append(cur)
            cur = word
        else:
            cur = cand
    if cur: lines.append(cur)
    return lines[:maxl]


def spinner(img, cx, cy, r, t, color, thick=2):
    a = (t * 300) % 360
    cv2.ellipse(img, (int(cx),int(cy)), (r,r), 0, int(a), int(a)+240,
                color, thick, cv2.LINE_AA)


# ─── object markers on camera frame ──────────────────────────────────────────

def pos_to_xy(pos: str, fw: int, fh: int) -> tuple:
    fx, fy = POSITION_MAP.get(pos.lower().strip(), (0.50, 0.50))
    return int(fw * fx), int(fh * fy)


def draw_object_markers(frame: np.ndarray, objects: list) -> np.ndarray:
    """Return a copy of the frame with a numbered circle at each object's position."""
    out = frame.copy()
    h, w = out.shape[:2]
    for i, obj in enumerate(objects):
        cx, cy = pos_to_xy(obj.get("position", "center"), w, h)
        color  = MARKER_COLORS[i % len(MARKER_COLORS)]
        r = 22
        cv2.circle(out, (cx,cy), r, (0,0,0), -1, cv2.LINE_AA)
        cv2.circle(out, (cx,cy), r, color,    2,  cv2.LINE_AA)
        n  = str(i+1)
        tw = txt_w(n, 0.55, 2)
        txt(out, n, cx-tw//2, cy+6, color, 0.55, 2)
        name = obj.get("name","?")[:14]
        lw   = txt_w(name, 0.38)
        lx, ly = cx-lw//2, cy+r+14
        rrect(out, lx-4, ly-12, lw+8, 16, 3, (0,0,0), alpha=0.72)
        txt(out, name, lx, ly, color, 0.38)
    return out


# ─── LLM prompts ──────────────────────────────────────────────────────────────

SCENE_PROMPT = """Analyse this image. Return ONLY valid JSON — no markdown, no explanation.

{
  "description": "Two sentences describing the scene.",
  "objects": [
    {
      "name": "object name",
      "position": "one of: left / center / right / top / bottom / top-left / top-right / bottom-left / bottom-right",
      "detail": "one short useful observation about this specific object"
    }
  ]
}

List every distinct visible object. Be concise and accurate."""

LOCATION_PROMPT = (
    "Look at the region around {x_pct}% from the left and {y_pct}% from the top "
    "of this image. What specific object or area is there? Describe it in 1-2 sentences."
)

FOLLOWUP_PROMPT = (
    "Focus only on the '{name}' in this image. "
    "Describe its appearance, colour, condition, and any notable features in 2-3 sentences."
)


# ─── LLM functions ────────────────────────────────────────────────────────────

def _strip_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$",        "", raw)
    return raw.strip()


def _parse_objects_fallback(text: str) -> list:
    """Regex fallback when model returns prose instead of JSON."""
    for pat in [r'\d+\.\s*([^\n,\.]{2,40})', r'[-•*]\s*([^\n]{2,40})']:
        found = re.findall(pat, text)
        if found:
            return [{"name": o.strip(), "position": "center", "detail": ""}
                    for o in found][:10]
    return []


def analyse_frame(model: str, b64: str) -> tuple:
    """
    Send image to the VLM, parse JSON response.
    Returns (description, objects_list). Never raises.

    Fix: raw_text is initialised BEFORE the try block so the except clause
    can reference it safely even when ollama.chat() itself throws.
    (Old version used `"r" in dir()` — unreliable and a code smell.)
    """
    raw_text = ""
    try:
        r        = ollama.chat(model=model, messages=[{
            "role":    "user",
            "content": SCENE_PROMPT,
            "images":  [b64],
        }])
        raw_text = r["message"]["content"]
        data     = json.loads(_strip_json(raw_text))
        desc     = data.get("description", "")
        objs     = []
        for o in data.get("objects", []):
            if isinstance(o, str): o = {"name": o}
            objs.append({
                "name":     str(o.get("name",     "?"))[:30],
                "position": str(o.get("position", "center")),
                "detail":   str(o.get("detail",   ""))[:80],
            })
        return desc, objs[:10]
    except json.JSONDecodeError:
        first = raw_text.split(".")[0].strip() if raw_text else "No description."
        return first, _parse_objects_fallback(raw_text)
    except Exception as e:
        return f"[Error: {e}]", []


def followup_query(model: str, b64: str, obj: dict) -> str:
    try:
        r = ollama.chat(model=model, messages=[{
            "role":    "user",
            "content": FOLLOWUP_PROMPT.format(name=obj["name"]),
            "images":  [b64],
        }])
        return r["message"]["content"].strip()
    except Exception as e:
        return f"[Error: {e}]"


def location_query(model: str, b64: str, x_pct: int, y_pct: int) -> str:
    """
    Ask what's at a specific (x%, y%) position in the image.
    Triggered when the user clicks directly on the camera feed.
    """
    try:
        r = ollama.chat(model=model, messages=[{
            "role":    "user",
            "content": LOCATION_PROMPT.format(x_pct=x_pct, y_pct=y_pct),
            "images":  [b64],
        }])
        return r["message"]["content"].strip()
    except Exception as e:
        return f"[Error: {e}]"


# ─── encoding ─────────────────────────────────────────────────────────────────

def encode(frame: np.ndarray, size=INF_SIZE) -> str:
    """Resize → JPEG → base64 string ready for the ollama API."""
    h, w = frame.shape[:2]
    s    = size / max(h, w)
    if s < 1.0:
        frame = cv2.resize(frame, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def make_thumb(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)


# ─── sidebar — split into one function per section ────────────────────────────

TAG_H = 28

def _tag_layout(objects, sx, start_y, avail_w):
    """Flow-layout: pack tags left-to-right, wrap when row is full."""
    rects, cx, cy = [], sx+12, start_y
    for i, obj in enumerate(objects):
        label = f"{i+1}  {obj['name'][:16]}"
        tw    = txt_w(label, 0.40) + 20
        if cx + tw > sx + avail_w - 12:
            cx = sx+12; cy += TAG_H+5
        rects.append((i, cx, cy, tw, TAG_H))
        cx += tw+6
    return rects


def _sb_header(canvas, st, sx, sw):
    canvas[0:58, sx:] = P["bg2"]
    cv2.line(canvas, (sx,58), (sx+sw,58), P["border"], 1)
    txt(canvas, "SnapAnnotator",           sx+14, 30, P["accent"], 0.62, 2, FB)
    txt(canvas, "local vision  •  offline", sx+16, 48, P["txt2"],  0.36)
    mdl = st["model"]
    mw  = txt_w(mdl, 0.36)
    rrect(canvas, sx+sw-mw-24, 17, mw+16, 20, 4, P["border"])
    txt(canvas, mdl, sx+sw-mw-16, 31, P["txt1"], 0.36)
    return 68


def _sb_status(canvas, st, sx, sw, cy, t, interval):
    phase = st["phase"]
    if st["busy"]:
        badge, bcol = "  ANALYSING", P["amber"]
    elif phase == "live":
        pulse = 0.45 + 0.55*math.sin(t*3.5)
        pcol  = tuple(int(c*pulse) for c in P["green"])
        cv2.circle(canvas, (sx+22, cy+11), 6, pcol, -1, cv2.LINE_AA)
        txt(canvas, "LIVE  —  press SPACE to capture",
            sx+34, cy+16, P["txt1"], 0.40)
        if st["auto_capture"]:
            remaining = max(0, interval - (time.time() - st["last_auto_time"]))
            lbl = f"AUTO  {remaining:.0f}s"
            aw  = txt_w(lbl, 0.38)+14
            rrect(canvas, sx+sw-aw-10, cy+2, aw, 20, 4,
                  P["auto"], alpha=0.25, bcolor=P["auto"])
            txt(canvas, lbl, sx+sw-aw-4, cy+16, P["auto"], 0.38)
        return cy+30
    elif phase == "annotated":
        badge, bcol = "  ANNOTATED", P["accent"]
    else:
        badge, bcol = "  CAPTURED",  P["amber"]
    bw = txt_w(badge, 0.44, 1)
    rrect(canvas, sx+12, cy, bw+22, 26, 6, bcol, alpha=0.18, bcolor=bcol)
    txt(canvas, badge, sx+22, cy+17, bcol, 0.44, 1, FB)
    return cy+34


def _sb_scene(canvas, st, sx, sw, cy, t):
    cv2.line(canvas, (sx+12,cy), (sx+sw-12,cy), P["border"], 1)
    cy += 12
    txt(canvas, "SCENE", sx+14, cy+11, P["txt2"], 0.36)
    cy += 16
    if st["busy"] and not st["description"]:
        ch = 58
        rrect(canvas, sx+10, cy, sw-20, ch, 8, P["bg2"], bcolor=P["border"])
        spinner(canvas, sx+sw//2, cy+ch//2, 14, t, P["accent"], 2)
        txt(canvas, "Analysing scene…",
            sx+sw//2-62, cy+ch//2+5, P["txt1"], 0.42)
        return cy+ch+10
    elif st["description"]:
        lines = wrap(st["description"], 38, 5)
        ch    = 14+len(lines)*19+8
        rrect(canvas, sx+10, cy, sw-20, ch, 8, P["bg2"], bcolor=P["border"])
        for i, ln in enumerate(lines):
            txt(canvas, ln, sx+18, cy+16+i*19, P["txt0"], 0.42)
        return cy+ch+10
    else:
        ch = 46
        rrect(canvas, sx+10, cy, sw-20, ch, 8, P["bg2"], bcolor=P["border"])
        txt(canvas, "No snapshot yet.",      sx+18, cy+18, P["txt2"], 0.42)
        txt(canvas, "Press SPACE to begin.", sx+18, cy+36, P["txt2"], 0.38)
        return cy+ch+10


def _sb_objects(canvas, st, sx, sw, cy, hover_idx, tag_out):
    cv2.line(canvas, (sx+12,cy), (sx+sw-12,cy), P["border"], 1)
    cy += 12
    txt(canvas, "OBJECTS  (click or press 1-9)", sx+14, cy+11, P["txt2"], 0.36)
    cy += 16
    if st["objects"]:
        trects     = _tag_layout(st["objects"], sx, cy, sw)
        tag_out[0] = trects
        for i, tx, ty, tw, th in trects:
            hov = (i == hover_idx)
            rrect(canvas, tx, ty, tw, th, 5,
                  P["accent"]  if hov else P["tag_bg"],
                  bcolor=P["accent"] if hov else P["tag_bdr"])
            txt(canvas, f"{i+1}  {st['objects'][i]['name'][:16]}",
                tx+10, ty+19,
                (10,10,10) if hov else P["txt0"], 0.40)
        if trects:
            cy = trects[-1][2]+TAG_H+8
        show = hover_idx if hover_idx >= 0 else st.get("selected_obj_idx", -1)
        if 0 <= show < len(st["objects"]):
            detail = st["objects"][show].get("detail","")
            if detail:
                dlines = wrap(detail, 38, 2)
                dh     = 10+len(dlines)*17+6
                rrect(canvas, sx+10, cy, sw-20, dh, 6, P["bg2"], bcolor=P["border"])
                for i, dl in enumerate(dlines):
                    txt(canvas, dl, sx+18, cy+14+i*17, P["txt1"], 0.39)
                cy += dh+6
        cy += 4
    elif st["busy"]:
        txt(canvas, "Detecting objects…", sx+14, cy+14, P["txt2"], 0.40); cy += 28
    else:
        txt(canvas, "No objects yet.",    sx+14, cy+14, P["txt2"], 0.40); cy += 28
    return cy


def _sb_followup(canvas, st, sx, sw, cy, t):
    cv2.line(canvas, (sx+12,cy), (sx+sw-12,cy), P["border"], 1)
    cy += 12
    txt(canvas, "FOLLOW-UP", sx+14, cy+11, P["txt2"], 0.36)
    cy += 16
    if st["busy"] and st["followup_obj"]:
        ch   = 50
        name = st["followup_obj"].get("name","location")
        lbl  = f"Asking about '{name}'..."
        rrect(canvas, sx+10, cy, sw-20, ch, 8, P["bg2"], bcolor=P["purple"])
        spinner(canvas, sx+sw//2, cy+ch//2, 12, t, P["purple"], 2)
        txt(canvas, lbl, sx+18, cy+ch//2+5, P["txt1"], 0.40)
        return cy+ch+10
    elif st["followup_answer"]:
        bname = st["followup_obj"].get("name","?") if st["followup_obj"] else "?"
        bw    = txt_w(bname, 0.40)+18
        rrect(canvas, sx+12, cy, bw, 24, 5, P["purple"], alpha=0.22, bcolor=P["purple"])
        txt(canvas, bname, sx+22, cy+16, P["purple"], 0.40)
        cy += 30
        alines = wrap(st["followup_answer"], 38, 6)
        ch     = 12+len(alines)*18+8
        rrect(canvas, sx+10, cy, sw-20, ch, 8, P["bg2"], bcolor=P["purple"])
        for i, al in enumerate(alines):
            txt(canvas, al, sx+18, cy+16+i*18, P["txt0"], 0.41)
        return cy+ch+10
    else:
        txt(canvas, "Click camera or a tag, or press 1-9.",
            sx+14, cy+14, P["txt2"], 0.40)
        return cy+28


def _sb_footer(canvas, st, sx, sw):
    fy = WIN_H-46
    canvas[fy:, sx:] = P["bg2"]
    cv2.line(canvas, (sx,fy), (sx+sw,fy), P["border"], 1)
    items = [
        ("SPC", "capture"),
        ("A",   "auto " + ("ON" if st["auto_capture"] else "off")),
        ("1-9", "query"), ("E","export"), ("Q","quit"),
    ]
    fx = sx+10
    for k, d in items:
        is_auto = k == "A" and st["auto_capture"]
        kc = P["auto"] if is_auto else P["border"]
        kw = txt_w(k, 0.38, 1)+10
        rrect(canvas, fx, fy+10, kw, 20, 3, kc, bcolor=kc)
        txt(canvas, k, fx+5, fy+24, P["txt0"], 0.38, 1, FB)
        fx += kw+5
        txt(canvas, d, fx, fy+24, P["auto"] if is_auto else P["txt2"], 0.36)
        fx += txt_w(d, 0.36)+8


def render_sidebar(canvas, st, hover_idx, t, tag_out, interval):
    sx, sw = CAM_W, SIDEBAR_W
    canvas[:, sx:] = P["bg1"]
    cv2.line(canvas, (sx,0), (sx,WIN_H), P["border"], 1)
    cy = _sb_header(canvas, st, sx, sw)
    cy = _sb_status(canvas, st, sx, sw, cy, t, interval)
    cy = _sb_scene( canvas, st, sx, sw, cy, t)
    cy = _sb_objects(canvas, st, sx, sw, cy, hover_idx, tag_out)
    _sb_followup(canvas, st, sx, sw, cy, t)
    _sb_footer(canvas, st, sx, sw)


# ─── camera overlays ──────────────────────────────────────────────────────────

def draw_cam_overlay(canvas, st, t):
    """Dim + scan-line animation while the AI is thinking."""
    if not st["busy"]: return
    fill(canvas, 0, 0, CAM_W, WIN_H, (0,0,0), alpha=0.35)
    sy = int(((t*0.4) % 1.0) * WIN_H)
    for dy in [0,1,2,-1,-2]:
        y = sy+dy
        if 0 <= y < WIN_H:
            cv2.line(canvas, (0,y), (CAM_W,y), P["accent"], 1)
    msg = "Analysing..."
    mw  = txt_w(msg, 0.6, 2)
    txt(canvas, msg, (CAM_W-mw)//2, WIN_H//2+6, P["accent"], 0.6, 2, FB)


def draw_history_strip(canvas, history):
    """
    Bottom-left corner: thumbnails of the last N annotated captures.
    Each thumbnail has an object-count badge — a quick timeline of past scenes.
    """
    if not history: return
    pad = 6
    y0  = WIN_H - THUMB_H - pad - 2
    for i, (thumb, n_obj) in enumerate(history):
        x0 = pad + i*(THUMB_W+pad)
        cv2.rectangle(canvas,(x0-1,y0-1),(x0+THUMB_W,y0+THUMB_H),P["border"],1)
        canvas[y0:y0+THUMB_H, x0:x0+THUMB_W] = thumb
        badge = str(n_obj)
        bw    = txt_w(badge, 0.35)+8
        rrect(canvas, x0+THUMB_W-bw-2, y0+2, bw, 14, 3, P["accent"], alpha=0.9)
        txt(canvas, badge, x0+THUMB_W-bw, y0+12, (10,10,10), 0.35)


def draw_auto_ring(canvas, st, t, interval):
    """
    Top-right of camera: circular progress arc counting down to next auto-capture.
    Fills completely just before a capture fires — satisfying to watch.
    """
    if not st["auto_capture"]: return
    progress = min((time.time()-st["last_auto_time"]) / interval, 1.0)
    cx, cy   = CAM_W-28, 28
    cv2.circle(canvas, (cx,cy), 18, (30,30,30), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx,cy), (16,16), -90, 0, int(progress*360),
                P["auto"], 2, cv2.LINE_AA)
    txt(canvas, "A", cx-5, cy+5, P["auto"], 0.40, 1, FB)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="SnapAnnotator — offline vision AI")
    ap.add_argument("--model",         "-m", default="moondream")
    ap.add_argument("--auto-interval", "-i", type=float, default=AUTO_INTERVAL,
                    help="Seconds between auto-captures (default 5)")
    args     = ap.parse_args()
    interval = args.auto_interval

    # ── preflight check ───────────────────────────────────────────────────────
    print("\n  Connecting to ollama...", end="", flush=True)
    try:
        available = [m["model"] for m in ollama.list().get("models", [])]
        if not any(args.model.split(":")[0] in m for m in available):
            print(f"\n\n  WARNING: '{args.model}' not found locally.")
            print(f"  Run:  ollama pull {args.model}\n")
        else:
            print(" OK")
    except Exception as e:
        print(f"\n  ERROR: can't reach ollama — {e}")
        print("  Make sure 'ollama serve' is running in another terminal.\n")
        sys.exit(1)

    # ── webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("  ERROR: cannot open webcam."); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    EXPORT_DIR.mkdir(exist_ok=True)

    # ── shared state ──────────────────────────────────────────────────────────
    st = dict(
        model            = args.model,
        phase            = "live",
        frame            = None,
        annotated        = None,
        b64              = None,
        description      = "",
        objects          = [],
        followup_obj     = None,
        followup_answer  = "",
        selected_obj_idx = -1,
        busy             = False,
        flash            = 0.0,
        auto_capture     = False,
        last_auto_time   = time.time(),
        history          = deque(maxlen=HISTORY_MAX),
    )

    ev               = queue.Queue()
    tag_rects_holder = [[]]
    mouse_pos        = [0, 0]

    # ── mouse callback ─────────────────────────────────────────────────────────
    def on_mouse(event, x, y, flags, _):
        mouse_pos[0], mouse_pos[1] = x, y
        if event != cv2.EVENT_LBUTTONDOWN: return
        with _lock:
            busy  = st["busy"]
            phase = st["phase"]
            b64   = st["b64"]
            frame = st["frame"]
        if busy: return

        # click on camera region → location query
        if x < CAM_W and phase == "annotated" and b64:
            fw    = frame.shape[1] if frame is not None else CAM_W
            fh    = frame.shape[0] if frame is not None else WIN_H
            x_pct = int(x / fw * 100)
            y_pct = int(y / fh * 100)
            print(f"\n  Click at ({x_pct}%, {y_pct}%) — querying location...")
            ev.put(("location_query", x_pct, y_pct))
            return

        # click on sidebar tag
        if x >= CAM_W and phase == "annotated":
            for i, tx, ty, tw, th in tag_rects_holder[0]:
                if tx <= x <= tx+tw and ty <= y <= ty+th:
                    ev.put(("query", i)); return

    cv2.namedWindow("SnapAnnotator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SnapAnnotator", WIN_W, WIN_H)
    cv2.setMouseCallback("SnapAnnotator", on_mouse)

    # ── capture helper ─────────────────────────────────────────────────────────
    def do_capture():
        ret, frame = cap.read()
        if not ret: return
        frame = cv2.flip(frame, 1)
        b64   = encode(frame)
        locked(st, frame=frame, b64=b64, phase="capturing",
               description="", objects=[], annotated=None,
               followup_obj=None, followup_answer="",
               selected_obj_idx=-1, busy=True, flash=1.0,
               last_auto_time=time.time())
        tag_rects_holder[0] = []
        print(f"\n  Captured. Sending to {args.model}...")
        threading.Thread(target=_worker_analyse, args=(b64,), daemon=True).start()

    # ── workers ───────────────────────────────────────────────────────────────
    def _worker_analyse(b64):
        t0 = time.time()
        desc, objs = analyse_frame(st["model"], b64)
        print(f"  Analysis done in {time.time()-t0:.1f}s  ({len(objs)} objects)")
        ev.put(("analysis_done", desc, objs))

    def _worker_followup(b64, obj):
        t0  = time.time()
        ans = followup_query(st["model"], b64, obj)
        print(f"  Follow-up done in {time.time()-t0:.1f}s")
        ev.put(("followup_done", obj, ans))

    def _worker_location(b64, x_pct, y_pct):
        t0  = time.time()
        ans = location_query(st["model"], b64, x_pct, y_pct)
        print(f"  Location query done in {time.time()-t0:.1f}s")
        ev.put(("location_done", x_pct, y_pct, ans))

    # ── main loop ──────────────────────────────────────────────────────────────
    print("  Webcam ready.  SPACE=capture  A=auto-capture  Q=quit\n")

    while True:
        t_now = time.time()

        if st["phase"] == "live":
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            with _lock: st["frame"] = frame

        # auto-capture trigger
        with _lock:
            auto_on = st["auto_capture"]
            busy    = st["busy"]
            phase   = st["phase"]
            since   = t_now - st["last_auto_time"]
        if auto_on and not busy and phase == "live" and since >= interval:
            do_capture()

        # drain event queue
        while not ev.empty():
            msg  = ev.get_nowait()
            kind = msg[0]

            if kind == "analysis_done":
                _, desc, objs = msg
                annotated = draw_object_markers(st["frame"], objs)
                thumb     = make_thumb(annotated)
                with _lock:
                    st["description"]     = desc
                    st["objects"]         = objs
                    st["annotated"]       = annotated
                    st["busy"]            = False
                    st["phase"]           = "annotated"
                    st["history"].appendleft((thumb, len(objs)))

            elif kind == "followup_done":
                _, obj, ans = msg
                locked(st, followup_obj=obj, followup_answer=ans, busy=False)

            elif kind == "location_done":
                _, x_pct, y_pct, ans = msg
                locked(st,
                       followup_obj={"name": f"{x_pct}%, {y_pct}%"},
                       followup_answer=ans, busy=False)

            elif kind == "query":
                i = msg[1]
                with _lock:
                    ok = 0 <= i < len(st["objects"]) and not st["busy"]
                    if ok:
                        obj = st["objects"][i]; b64 = st["b64"]
                        st["selected_obj_idx"] = i
                        st["followup_answer"]  = ""
                        st["followup_obj"]     = obj
                        st["busy"]             = True
                if ok:
                    threading.Thread(target=_worker_followup,
                                     args=(b64, obj), daemon=True).start()

            elif kind == "location_query":
                _, x_pct, y_pct = msg
                with _lock:
                    ok  = not st["busy"]; b64 = st["b64"]
                    if ok:
                        st["busy"]            = True
                        st["followup_answer"] = ""
                        st["followup_obj"]    = {"name": "location"}
                if ok:
                    threading.Thread(target=_worker_location,
                                     args=(b64, x_pct, y_pct), daemon=True).start()

        # ── compose canvas ────────────────────────────────────────────────────
        canvas = np.full((WIN_H, WIN_W, 3), P["bg0"], dtype=np.uint8)

        with _lock:
            cam_src = (st["annotated"] if st["phase"] == "annotated"
                       and st["annotated"] is not None else st["frame"])
            history = list(st["history"])
            flash   = st["flash"]

        if cam_src is not None:
            ch, cw = cam_src.shape[:2]
            scale  = min(CAM_W/cw, WIN_H/ch)
            dw, dh = int(cw*scale), int(ch*scale)
            ox, oy = (CAM_W-dw)//2, (WIN_H-dh)//2
            canvas[oy:oy+dh, ox:ox+dw] = cv2.resize(cam_src, (dw,dh))

        draw_cam_overlay(canvas, st, t_now)
        draw_history_strip(canvas, history)
        draw_auto_ring(canvas, st, t_now, interval)

        if flash > 0:
            fill(canvas, 0, 0, CAM_W, WIN_H, P["flash"], alpha=flash*0.85)
            with _lock: st["flash"] = max(0.0, st["flash"]-0.1)

        if st["phase"] == "live":
            pulse = 0.4 + 0.6*math.sin(t_now*4)
            dc    = tuple(int(c*pulse) for c in P["green"])
            cv2.circle(canvas, (18,18), 6, dc, -1, cv2.LINE_AA)
            txt(canvas, "LIVE", 30, 23, P["green"], 0.42)

        hover_idx = -1
        mx, my    = mouse_pos
        for i, tx, ty, tw, th in tag_rects_holder[0]:
            if tx <= mx <= tx+tw and ty <= my <= ty+th:
                hover_idx = i; break

        render_sidebar(canvas, st, hover_idx, t_now, tag_rects_holder, interval)
        cv2.imshow("SnapAnnotator", canvas)

        # ── keys ──────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        elif key == ord(' '):
            if st["phase"] == "live":
                do_capture()
            else:
                locked(st, phase="live", description="", objects=[],
                       annotated=None, followup_obj=None,
                       followup_answer="", busy=False, selected_obj_idx=-1)
                tag_rects_holder[0] = []

        elif key == ord('a'):
            with _lock:
                st["auto_capture"]   = not st["auto_capture"]
                st["last_auto_time"] = time.time()
            print(f"  Auto-capture {'ON' if st['auto_capture'] else 'OFF'}"
                  f"  ({interval}s interval)")

        elif key == ord('c'):
            locked(st, phase="live", description="", objects=[],
                   annotated=None, followup_obj=None,
                   followup_answer="", busy=False, selected_obj_idx=-1)
            tag_rects_holder[0] = []

        elif key == ord('e') and st["annotated"] is not None:
            ts   = datetime.now().strftime("%H%M%S")
            path = EXPORT_DIR / f"snap_{ts}.jpg"
            cv2.imwrite(str(path), st["annotated"])
            print(f"  Exported → {path}")

        elif ord('1') <= key <= ord('9'):
            ev.put(("query", key - ord('1')))

    # ── cleanup ────────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("\n  Closed. See you at Day 13!\n")


if __name__ == "__main__":
    main()
