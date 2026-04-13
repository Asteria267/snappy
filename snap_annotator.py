"""
SnapAnnotator  —  Day 12  Enhanced Edition
===========================================
SPACE  — capture & analyse
1–9    — query object by number (faster than clicking)
E      — export annotated frame to ./exports/
C      — clear / return to live
Q/ESC  — quit

Mouse: click any object tag in the sidebar to query it.

Prerequisites:
    pip install opencv-python pillow ollama numpy
    ollama pull moondream
    ollama serve          ← separate terminal
"""

import argparse, base64, io, json, math, queue
import re, sys, threading, time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import ollama

# ─── layout ──────────────────────────────────────────────────────────────────
CAM_W      = 1280
CAM_H      = 720
SIDEBAR_W  = 380
WIN_W      = CAM_W + SIDEBAR_W
WIN_H      = CAM_H
INF_SIZE   = 512          # resize before sending to model
EXPORT_DIR = Path("exports")

# ─── palette (BGR) ───────────────────────────────────────────────────────────
P = {
    "bg0":      ( 10,  10,  18),   # window bg
    "bg1":      ( 18,  20,  30),   # sidebar bg
    "bg2":      ( 26,  28,  42),   # card bg
    "border":   ( 48,  52,  78),   # card border
    "accent":   (210, 190,   0),   # cyan-yellow
    "green":    ( 50, 200,  80),
    "amber":    ( 30, 165, 230),
    "purple":   (200,  90, 180),
    "red":      ( 70,  70, 220),
    "txt0":     (240, 240, 250),   # primary text
    "txt1":     (155, 160, 185),   # secondary
    "txt2":     ( 80,  86, 112),   # dim
    "tag_bg":   ( 30,  36,  56),
    "tag_bdr":  ( 65,  75, 115),
    "flash":    (255, 255, 255),
}

# per-object marker colours (drawn on camera feed)
MARKER_COLORS = [
    (  0, 220, 255), (  0, 200, 100), (220, 100,   0),
    (200,   0, 220), (  0, 180, 220), (100, 220,   0),
    (220, 180,   0), (  0, 120, 255),
]

F  = cv2.FONT_HERSHEY_SIMPLEX
FB = cv2.FONT_HERSHEY_DUPLEX


# ─── drawing helpers ──────────────────────────────────────────────────────────

def fill(img, x, y, w, h, color, alpha=1.0):
    x, y, w, h = int(x), int(y), int(w), int(h)
    sub = img[y:y+h, x:x+w]
    if sub.shape[0] == 0 or sub.shape[1] == 0: return
    if alpha >= 1.0:
        img[y:y+h, x:x+w] = color
    else:
        blended = np.full_like(sub, color)
        cv2.addWeighted(blended, alpha, sub, 1.0-alpha, 0, sub)
        img[y:y+h, x:x+w] = sub


def rrect(img, x, y, w, h, r, color, alpha=1.0, bcolor=None, bthick=1):
    """Rounded filled rectangle, optional border."""
    x,y,w,h,r = int(x),int(y),int(w),int(h),min(int(r),w//2,h//2)
    if alpha < 1.0:
        ov = img.copy()
        _rr_solid(ov, x, y, w, h, r, color)
        cv2.addWeighted(ov, alpha, img, 1.0-alpha, 0, img)
    else:
        _rr_solid(img, x, y, w, h, r, color)
    if bcolor:
        _rr_border(img, x, y, w, h, r, bcolor, bthick)


def _rr_solid(img, x, y, w, h, r, c):
    cv2.rectangle(img, (x+r,y),   (x+w-r,y+h),   c, -1)
    cv2.rectangle(img, (x,y+r),   (x+w,y+h-r),   c, -1)
    for cx,cy in [(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
        cv2.circle(img, (cx,cy), r, c, -1)


def _rr_border(img, x, y, w, h, r, c, t):
    cv2.line(img,(x+r,y),(x+w-r,y),c,t)
    cv2.line(img,(x+r,y+h),(x+w-r,y+h),c,t)
    cv2.line(img,(x,y+r),(x,y+h-r),c,t)
    cv2.line(img,(x+w,y+r),(x+w,y+h-r),c,t)
    for (cx,cy,a) in [(x+r,y+r,180),(x+w-r,y+r,270),(x+r,y+h-r,90),(x+w-r,y+h-r,0)]:
        cv2.ellipse(img,(cx,cy),(r,r),0,a,a+90,c,t)


def txt(img, text, x, y, color, scale=0.46, thick=1, font=F):
    cv2.putText(img, str(text), (int(x),int(y)), font, scale, color, thick, cv2.LINE_AA)


def txt_w(text, scale=0.46, thick=1):
    (w,_),_ = cv2.getTextSize(str(text), F, scale, thick)
    return w


def wrap(text, maxc, maxl=99):
    lines, cur = [], ""
    for w in str(text).split():
        cand = (cur+" "+w).strip()
        if len(cand) > maxc:
            if cur: lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur: lines.append(cur)
    return lines[:maxl]


def spinner(img, cx, cy, r, t, color, thick=2):
    a = (t*300) % 360
    cv2.ellipse(img,(int(cx),int(cy)),(r,r),0,int(a),int(a)+240,color,thick,cv2.LINE_AA)


# ─── position text → camera pixel coords ─────────────────────────────────────

def pos_to_xy(pos: str, fw: int, fh: int):
    """Map 'top-left', 'center', etc. → (cx, cy) on the camera frame."""
    p = pos.lower()
    x = fw//4 if "left" in p else (3*fw//4 if "right" in p else fw//2)
    y = fh//4 if "top"  in p else (3*fh//4 if "bottom" in p else fh//2)
    return x, y


def draw_object_markers(frame, objects: list):
    """
    Draw numbered circles on the camera frame at approximate object positions.
    Returns a copy — does not mutate the original.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    for i, obj in enumerate(objects):
        cx, cy = pos_to_xy(obj.get("position","center"), w, h)
        color  = MARKER_COLORS[i % len(MARKER_COLORS)]
        r = 22
        # filled circle with border
        cv2.circle(out, (cx,cy), r,   (0,0,0),  -1, cv2.LINE_AA)
        cv2.circle(out, (cx,cy), r,   color,      2, cv2.LINE_AA)
        # number
        n = str(i+1)
        tw = txt_w(n, 0.55, 2)
        txt(out, n, cx-tw//2, cy+6, color, 0.55, 2)
        # short name label
        name = obj.get("name","?")[:14]
        lw   = txt_w(name, 0.38)
        lx, ly = cx - lw//2, cy + r + 14
        rrect(out, lx-4, ly-12, lw+8, 16, 3, (0,0,0), alpha=0.72)
        txt(out, name, lx, ly, color, 0.38)
    return out


# ─── LLM ─────────────────────────────────────────────────────────────────────

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


def _strip_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$",        "", raw)
    return raw.strip()


def _parse_objects_fallback(text: str) -> list:
    """Last-resort parser when moondream returns plain text instead of JSON."""
    objs = []
    for pat in [r'\d+\.\s*([^\n,\.]{2,40})', r'[-•*]\s*([^\n]{2,40})']:
        found = re.findall(pat, text)
        if found:
            objs = [{"name":o.strip(),"position":"center","detail":""} for o in found]
            break
    return objs[:10]


def analyse_frame(model: str, b64: str) -> tuple[str, list]:
    """Single LLM call → (description, objects list). Never raises."""
    try:
        r = ollama.chat(model=model, messages=[{
            "role":    "user",
            "content": SCENE_PROMPT,
            "images":  [b64],
        }])
        raw  = _strip_json(r["message"]["content"])
        data = json.loads(raw)
        desc = data.get("description", "")
        objs = data.get("objects", [])
        # normalise — ensure each has name/position/detail
        clean = []
        for o in objs:
            if isinstance(o, str): o = {"name": o}
            clean.append({
                "name":     str(o.get("name","?"))[:30],
                "position": str(o.get("position","center")),
                "detail":   str(o.get("detail",""))[:80],
            })
        return desc, clean[:10]
    except json.JSONDecodeError:
        # moondream returned prose — extract what we can
        raw_text = r["message"]["content"] if "r" in dir() else ""
        objs = _parse_objects_fallback(raw_text)
        # try to pull description: first sentence
        first = raw_text.split(".")[0].strip() if raw_text else "No description."
        return first, objs
    except Exception as e:
        return f"[Error: {e}]", []


def followup_query(model: str, b64: str, obj: dict) -> str:
    """Ask a follow-up question about a specific object."""
    try:
        r = ollama.chat(model=model, messages=[{
            "role":    "user",
            "content": (
                f"Look only at the '{obj['name']}' in this image. "
                f"Describe its appearance, colour, condition, and any notable "
                f"features in 2–3 sentences."
            ),
            "images":  [b64],
        }])
        return r["message"]["content"].strip()
    except Exception as e:
        return f"[Error: {e}]"


# ─── frame encoding ───────────────────────────────────────────────────────────

def encode(frame: np.ndarray, size=INF_SIZE) -> str:
    h, w = frame.shape[:2]
    s    = size / max(h, w)
    if s < 1.0:
        frame = cv2.resize(frame, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ─── sidebar renderer ─────────────────────────────────────────────────────────

TAG_H = 28

def _tag_layout(objects, sx, start_y, avail_w):
    """Flow-layout: returns list of (i, x, y, w, h)."""
    rects = []
    cx, cy = sx+12, start_y
    for i, obj in enumerate(objects):
        label = f"{i+1}  {obj['name'][:16]}"
        tw    = txt_w(label, 0.40) + 20
        if cx + tw > sx + avail_w - 12:
            cx = sx + 12
            cy += TAG_H + 5
        rects.append((i, cx, cy, tw, TAG_H))
        cx += tw + 6
    return rects


def render_sidebar(canvas, st, hover_idx, t, tag_out):
    """
    Draw the sidebar onto canvas in-place.
    Writes computed tag_rects into tag_out[0] so caller can use them for hit-testing.
    """
    sx = CAM_W
    sw = SIDEBAR_W

    # background
    canvas[:, sx:] = P["bg1"]
    cv2.line(canvas, (sx,0), (sx,WIN_H), P["border"], 1)

    cy = 0

    # ── header ────────────────────────────────────────────────────────────────
    canvas[0:58, sx:] = P["bg2"]
    cv2.line(canvas, (sx,58), (sx+sw,58), P["border"], 1)
    txt(canvas, "SnapAnnotator", sx+14, 30, P["accent"], 0.62, 2, FB)
    txt(canvas, "local vision  •  offline", sx+16, 48, P["txt2"], 0.36)
    # model badge
    mdl = st["model"]
    mw  = txt_w(mdl, 0.36)
    rrect(canvas, sx+sw-mw-24, 17, mw+16, 20, 4, P["border"])
    txt(canvas, mdl, sx+sw-mw-16, 31, P["txt1"], 0.36)
    cy = 68

    # ── status badge ──────────────────────────────────────────────────────────
    phase = st["phase"]
    if st["busy"]:
        badge, bcol = "⟳  ANALYSING", P["amber"]
    elif phase == "live":
        # pulsing dot
        pulse = 0.45 + 0.55*math.sin(t*3.5)
        pcol  = tuple(int(c*pulse) for c in P["green"])
        cv2.circle(canvas, (sx+22, cy+11), 6, pcol, -1, cv2.LINE_AA)
        txt(canvas, "LIVE  —  press SPACE to capture", sx+34, cy+16, P["txt1"], 0.40)
        cy += 30
        badge, bcol = None, None
    elif phase == "annotated":
        badge, bcol = "✓  ANNOTATED", P["accent"]
    else:
        badge, bcol = "■  CAPTURED", P["amber"]

    if badge:
        bw = txt_w(badge, 0.44, 1)
        rrect(canvas, sx+12, cy, bw+22, 26, 6, bcol, alpha=0.18,
              bcolor=bcol, bthick=1)
        txt(canvas, badge, sx+22, cy+17, bcol, 0.44, 1, FB)
        cy += 34

    # divider helper
    def div(y):
        cv2.line(canvas,(sx+12,y),(sx+sw-12,y),P["border"],1)

    div(cy); cy += 12

    # ── scene description ─────────────────────────────────────────────────────
    txt(canvas, "SCENE", sx+14, cy+11, P["txt2"], 0.36)
    cy += 16

    if st["busy"] and not st["description"]:
        ch = 58
        rrect(canvas, sx+10, cy, sw-20, ch, 8, P["bg2"], bcolor=P["border"])
        spinner(canvas, sx+sw//2, cy+ch//2, 14, t, P["accent"], 2)
        txt(canvas, "Analysing scene…", sx+sw//2-62, cy+ch//2+5, P["txt1"], 0.42)
        cy += ch + 10
    elif st["description"]:
        lines  = wrap(st["description"], 38, 5)
        ch     = 14 + len(lines)*19 + 8
        rrect(canvas, sx+10, cy, sw-20, ch, 8, P["bg2"], bcolor=P["border"])
        for i, ln in enumerate(lines):
            txt(canvas, ln, sx+18, cy+16+i*19, P["txt0"], 0.42)
        cy += ch + 10
    else:
        ch = 46
        rrect(canvas, sx+10, cy, sw-20, ch, 8, P["bg2"], bcolor=P["border"])
        txt(canvas, "No snapshot yet.", sx+18, cy+18, P["txt2"], 0.42)
        txt(canvas, "Press SPACE to begin.", sx+18, cy+36, P["txt2"], 0.38)
        cy += ch + 10

    div(cy); cy += 12

    # ── objects ───────────────────────────────────────────────────────────────
    txt(canvas, "OBJECTS  (click or press 1–9)", sx+14, cy+11, P["txt2"], 0.36)
    cy += 16

    if st["objects"]:
        trects = _tag_layout(st["objects"], sx, cy, sw)
        tag_out[0] = trects   # ← hand back to caller for hit-testing

        for i, tx, ty, tw, th in trects:
            is_hov = (i == hover_idx)
            bg  = P["accent"] if is_hov else P["tag_bg"]
            bdr = P["accent"] if is_hov else P["tag_bdr"]
            tc_ = (10,10,10) if is_hov else P["txt0"]
            rrect(canvas, tx, ty, tw, th, 5, bg, bcolor=bdr)
            label = f"{i+1}  {st['objects'][i]['name'][:16]}"
            txt(canvas, label, tx+10, ty+19, tc_, 0.40)

        if trects:
            cy = trects[-1][2] + TAG_H + 8

        # detail line for hovered / selected object
        sel = st.get("selected_obj_idx", -1)
        show_idx = hover_idx if hover_idx >= 0 else sel
        if 0 <= show_idx < len(st["objects"]):
            detail = st["objects"][show_idx].get("detail","")
            if detail:
                dlines = wrap(detail, 38, 2)
                dh = 10 + len(dlines)*17 + 6
                rrect(canvas, sx+10, cy, sw-20, dh, 6, P["bg2"], bcolor=P["border"])
                for i,dl in enumerate(dlines):
                    txt(canvas, dl, sx+18, cy+14+i*17, P["txt1"], 0.39)
                cy += dh + 6

        cy += 4
    elif st["busy"]:
        txt(canvas, "Detecting objects…", sx+14, cy+14, P["txt2"], 0.40)
        cy += 28
    else:
        txt(canvas, "No objects yet.", sx+14, cy+14, P["txt2"], 0.40)
        cy += 28

    div(cy); cy += 12

    # ── follow-up ─────────────────────────────────────────────────────────────
    txt(canvas, "FOLLOW-UP", sx+14, cy+11, P["txt2"], 0.36)
    cy += 16

    if st["busy"] and st["followup_obj"]:
        ch = 50
        rrect(canvas, sx+10, cy, sw-20, ch, 8, P["bg2"], bcolor=P["purple"])
        spinner(canvas, sx+sw//2, cy+ch//2, 12, t, P["purple"], 2)
        lbl = f"Asking about '{st['followup_obj']['name']}'…"
        txt(canvas, lbl, sx+18, cy+ch//2+5, P["txt1"], 0.40)
        cy += ch + 10
    elif st["followup_answer"]:
        # object badge
        bname = st["followup_obj"]["name"] if st["followup_obj"] else "?"
        bw = txt_w(bname, 0.40)+18
        rrect(canvas, sx+12, cy, bw, 24, 5, P["purple"], alpha=0.22,
              bcolor=P["purple"])
        txt(canvas, bname, sx+22, cy+16, P["purple"], 0.40)
        cy += 30
        alines = wrap(st["followup_answer"], 38, 6)
        ch = 12 + len(alines)*18 + 8
        rrect(canvas, sx+10, cy, sw-20, ch, 8, P["bg2"], bcolor=P["purple"])
        for i,al in enumerate(alines):
            txt(canvas, al, sx+18, cy+16+i*18, P["txt0"], 0.41)
        cy += ch + 10
    else:
        txt(canvas, "Click a tag or press 1–9.", sx+14, cy+14, P["txt2"], 0.40)
        cy += 28

    # ── footer ────────────────────────────────────────────────────────────────
    fy = WIN_H - 46
    canvas[fy:, sx:] = P["bg2"]
    cv2.line(canvas,(sx,fy),(sx+sw,fy),P["border"],1)

    shortcuts = [("SPACE","capture"), ("1–9","query obj"), ("E","export"), ("Q","quit")]
    fx = sx+12
    for k, d in shortcuts:
        kw = txt_w(k, 0.38, 1)+10
        rrect(canvas, fx, fy+10, kw, 20, 3, P["border"])
        txt(canvas, k, fx+5, fy+24, P["txt0"], 0.38, 1, FB)
        fx += kw+6
        txt(canvas, d, fx, fy+24, P["txt2"], 0.36)
        fw = txt_w(d, 0.36)+10
        fx += fw+6


# ─── camera overlay during analysis ──────────────────────────────────────────

def draw_cam_overlay(canvas, st, t):
    """Scan-line + dim overlay on the camera region when analysing."""
    if not st["busy"]: return
    # dim
    fill(canvas, 0, 0, CAM_W, WIN_H, (0,0,0), alpha=0.35)
    # scan line
    sy = int(((t*0.4) % 1.0) * WIN_H)
    for dy, a in [(0,0.6),(1,0.3),(2,0.1),(-1,0.3),(-2,0.1)]:
        y = sy+dy
        if 0 <= y < WIN_H:
            cv2.line(canvas, (0,y), (CAM_W,y), P["accent"], 1)
    # label
    msg = "Analysing…"
    mw  = txt_w(msg, 0.6, 2)
    txt(canvas, msg, (CAM_W-mw)//2, WIN_H//2+6, P["accent"], 0.6, 2, FB)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model","-m", default="moondream")
    args = ap.parse_args()

    # preflight
    print("  Connecting to ollama…", end="", flush=True)
    try:
        available = [m["model"] for m in ollama.list().get("models",[])]
        base = args.model.split(":")[0]
        if not any(base in m for m in available):
            print(f"\n\n  WARNING: '{args.model}' not found locally.")
            print(f"  Run:  ollama pull {args.model}\n")
        else:
            print(" OK")
    except Exception as e:
        print(f"\n  ERROR: can't reach ollama. Run 'ollama serve'\n  {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ERROR: cannot open webcam."); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    EXPORT_DIR.mkdir(exist_ok=True)

    # ── app state ─────────────────────────────────────────────────────────────
    st = dict(
        model         = args.model,
        phase         = "live",    # live | capturing | annotated
        frame         = None,      # last captured frame (raw)
        annotated     = None,      # frame with markers drawn
        b64           = None,
        description   = "",
        objects       = [],        # [{"name","position","detail"}]
        followup_obj  = None,
        followup_answer = "",
        selected_obj_idx = -1,
        busy          = False,
        flash         = 0.0,
    )

    # events from worker threads — main thread applies them
    ev = queue.Queue()
    tag_rects_holder = [[] ]      # mutable container so render_sidebar can write back
    mouse_pos = [0, 0]

    def on_mouse(event, x, y, flags, _):
        mouse_pos[0], mouse_pos[1] = x, y
        if event != cv2.EVENT_LBUTTONDOWN: return
        if st["phase"] != "annotated" or st["busy"]: return
        for i, tx, ty, tw, th in tag_rects_holder[0]:
            if tx <= x <= tx+tw and ty <= y <= ty+th:
                ev.put(("query", i)); return

    cv2.namedWindow("SnapAnnotator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SnapAnnotator", WIN_W, WIN_H)
    cv2.setMouseCallback("SnapAnnotator", on_mouse)

    def worker_analyse(b64):
        t0 = time.time()
        desc, objs = analyse_frame(st["model"], b64)
        print(f"  Analysis done in {time.time()-t0:.1f}s  ({len(objs)} objects)")
        ev.put(("analysis_done", desc, objs))

    def worker_followup(b64, obj):
        t0  = time.time()
        ans = followup_query(st["model"], b64, obj)
        print(f"  Follow-up done in {time.time()-t0:.1f}s")
        ev.put(("followup_done", obj, ans))

    print("  Webcam open.  Press SPACE to capture.\n")

    while True:
        t_now = time.time()

        # ── camera feed ───────────────────────────────────────────────────────
        if st["phase"] == "live":
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            st["frame"] = frame

        # ── process events ────────────────────────────────────────────────────
        while not ev.empty():
            msg = ev.get_nowait()
            kind = msg[0]

            if kind == "analysis_done":
                _, desc, objs = msg
                st["description"] = desc
                st["objects"]     = objs
                st["annotated"]   = draw_object_markers(st["frame"], objs)
                st["busy"]        = False
                st["phase"]       = "annotated"

            elif kind == "followup_done":
                _, obj, ans = msg
                st["followup_obj"]    = obj
                st["followup_answer"] = ans
                st["busy"]            = False

            elif kind == "query":
                i = msg[1]
                if 0 <= i < len(st["objects"]) and not st["busy"]:
                    st["selected_obj_idx"] = i
                    st["followup_answer"]  = ""
                    st["followup_obj"]     = st["objects"][i]
                    st["busy"]             = True
                    threading.Thread(
                        target=worker_followup,
                        args=(st["b64"], st["objects"][i]),
                        daemon=True
                    ).start()

        # ── compose frame ─────────────────────────────────────────────────────
        canvas = np.full((WIN_H, WIN_W, 3), P["bg0"], dtype=np.uint8)

        # camera region
        cam_src = (st["annotated"] if st["phase"]=="annotated" and st["annotated"] is not None
                   else st["frame"])
        if cam_src is not None:
            ch, cw = cam_src.shape[:2]
            scale  = min(CAM_W/cw, WIN_H/ch)
            dw,dh  = int(cw*scale), int(ch*scale)
            ox,oy  = (CAM_W-dw)//2, (WIN_H-dh)//2
            canvas[oy:oy+dh, ox:ox+dw] = cv2.resize(cam_src,(dw,dh))

        draw_cam_overlay(canvas, st, t_now)

        # flash on capture
        if st["flash"] > 0:
            fill(canvas, 0, 0, CAM_W, WIN_H, P["flash"], alpha=st["flash"]*0.85)
            st["flash"] = max(0.0, st["flash"]-0.1)

        # LIVE dot
        if st["phase"] == "live":
            pulse = 0.4 + 0.6*math.sin(t_now*4)
            dc = tuple(int(c*pulse) for c in P["green"])
            cv2.circle(canvas, (18,18), 6, dc, -1, cv2.LINE_AA)
            txt(canvas,"LIVE",30,23,P["green"],0.42)

        # sidebar
        hover_idx = -1
        for i,tx,ty,tw,th in tag_rects_holder[0]:
            mx,my = mouse_pos
            if tx<=mx<=tx+tw and ty<=my<=ty+th:
                hover_idx=i; break

        render_sidebar(canvas, st, hover_idx, t_now, tag_rects_holder)

        cv2.imshow("SnapAnnotator", canvas)

        # ── keys ──────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        elif key == ord(' '):
            if st["phase"] == "live":
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame,1)
                    st["frame"]           = frame
                    st["b64"]             = encode(frame)
                    st["phase"]           = "capturing"
                    st["description"]     = ""
                    st["objects"]         = []
                    st["annotated"]       = None
                    st["followup_obj"]    = None
                    st["followup_answer"] = ""
                    st["selected_obj_idx"]= -1
                    st["busy"]            = True
                    st["flash"]           = 1.0
                    tag_rects_holder[0]   = []
                    print(f"\n  Captured frame. Sending to {args.model}…")
                    threading.Thread(
                        target=worker_analyse,
                        args=(st["b64"],),
                        daemon=True
                    ).start()
            else:
                # reset to live
                st.update(phase="live", description="", objects=[],
                          annotated=None, followup_obj=None,
                          followup_answer="", busy=False, selected_obj_idx=-1)
                tag_rects_holder[0] = []

        elif key == ord('c'):
            st.update(phase="live", description="", objects=[],
                      annotated=None, followup_obj=None,
                      followup_answer="", busy=False, selected_obj_idx=-1)
            tag_rects_holder[0] = []

        elif key == ord('e') and st["annotated"] is not None:
            ts   = datetime.now().strftime("%H%M%S")
            path = EXPORT_DIR / f"snap_{ts}.jpg"
            cv2.imwrite(str(path), st["annotated"])
            print(f"  Exported → {path}")

        elif ord('1') <= key <= ord('9'):
            i = key - ord('1')
            ev.put(("query", i))

    cap.release()
    cv2.destroyAllWindows()
    print("\n  Closed.\n")


if __name__ == "__main__":
    main()
