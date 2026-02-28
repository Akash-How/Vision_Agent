import asyncio
import contextlib
import json
import os
import re
import time
from datetime import datetime, timezone
from fractions import Fraction

import av
import cv2
from aiortc import VideoStreamTrack
from vision_agents.plugins import openai
from vision_agents.testing import TestSession

try:
    # Preferred import if the package exposes it directly.
    from vision_agents import VisionAgent  # type: ignore
except Exception:
    # Fallback for current SDK layout.
    from vision_agents.core import Agent as VisionAgent  # type: ignore


AGENT_PROMPT = (
    "You are a realtime attention monitoring vision agent. Analyze the camera stream "
    "and determine if the user is Focused, Fatigued, or Distracted. "
    "Do not classify Fatigued from resting/neutral face alone. "
    "Require clear drowsiness signs such as prolonged eye closure, repeated yawning, or head nodding. "
    "Respond with JSON only: {\"state\":\"Focused|Fatigued|Distracted\","
    "\"confidence\":0.0-1.0,\"reason\":\"short reason\"}."
)
MODEL_NAME = os.environ.get("VISION_AGENT_MODEL", "minimax-m2:cloud")
API_BASE_URL = (
    os.environ.get("OPENAI_BASE_URL")
    or os.environ.get("VISION_AGENT_BASE_URL")
    or "https://api.ollama.com/v1"
)
API_KEY = (
    os.environ.get("OPENAI_API_KEY")
    or os.environ.get("VISION_AGENT_API_KEY")
    or os.environ.get("OLLAMA_API_KEY")
)
STREAM_FPS = 5
INFERENCE_INTERVAL_SECONDS = float(
    os.environ.get("VISION_AGENT_INFERENCE_INTERVAL", "1.0")
)
MAX_CONVERSATION_MESSAGES = int(os.environ.get("VISION_AGENT_MAX_HISTORY", "4"))
NO_FACE_CONSEC_FRAMES = int(os.environ.get("VISION_AGENT_NO_FACE_FRAMES", "18"))
FACE_DETECT_EVERY_N = int(os.environ.get("VISION_AGENT_FACE_DETECT_EVERY", "2"))
MIN_FACE_AREA_RATIO = float(os.environ.get("VISION_AGENT_MIN_FACE_AREA_RATIO", "0.006"))
FACE_MIN_SIZE_RATIO = float(os.environ.get("VISION_AGENT_MIN_SIZE_RATIO", "0.08"))
FACE_FRONTAL_MIN_NEIGHBORS = int(
    os.environ.get("VISION_AGENT_FACE_MIN_NEIGHBORS", "4")
)
FACE_PROFILE_ENABLED = (
    os.environ.get("VISION_AGENT_ENABLE_PROFILE_FACE", "1").strip().lower()
    not in {"0", "false", "no"}
)
FACE_MISS_CYCLES_TOLERANCE = int(
    os.environ.get("VISION_AGENT_FACE_MISS_TOLERANCE", "5")
)
FACE_ASPECT_MIN = float(os.environ.get("VISION_AGENT_FACE_ASPECT_MIN", "0.72"))
FACE_ASPECT_MAX = float(os.environ.get("VISION_AGENT_FACE_ASPECT_MAX", "1.50"))
FACE_CENTER_MAX_Y_RATIO = float(
    os.environ.get("VISION_AGENT_FACE_CENTER_MAX_Y_RATIO", "0.88")
)
FACE_REQUIRE_EYES = (
    os.environ.get("VISION_AGENT_FACE_REQUIRE_EYES", "1").strip().lower()
    not in {"0", "false", "no"}
)
WINDOW_NAME = "CogniGuard Vision Agent Session"
CAMERA_INDEX = int(os.environ.get("VISION_AGENT_CAMERA_INDEX", "0"))
CAMERA_BACKEND = os.environ.get("VISION_AGENT_CAMERA_BACKEND", "auto").strip().lower()
TRACKER_COLOR = (100, 200, 255)
TRACKER_THICKNESS = 3
TRACKER_POS_SMOOTH_FACTOR = float(
    os.environ.get("VISION_AGENT_TRACKER_POS_SMOOTH", "0.90")
)
TRACKER_SIZE_SMOOTH_FACTOR = float(
    os.environ.get("VISION_AGENT_TRACKER_SIZE_SMOOTH", "0.88")
)
TRACKER_DEADZONE_PX = float(
    os.environ.get("VISION_AGENT_TRACKER_DEADZONE_PX", "2.5")
)
TRACKER_MAX_STEP_PX = float(
    os.environ.get("VISION_AGENT_TRACKER_MAX_STEP_PX", "24.0")
)
FATIGUE_MIN_CONFIDENCE = float(
    os.environ.get("VISION_AGENT_FATIGUE_MIN_CONFIDENCE", "0.88")
)
FATIGUE_CONFIRM_STREAK = int(
    os.environ.get("VISION_AGENT_FATIGUE_CONFIRM_STREAK", "3")
)
FATIGUE_RECOVER_STREAK = int(
    os.environ.get("VISION_AGENT_FATIGUE_RECOVER_STREAK", "2")
)
FATIGUE_REQUIRE_REASON_KEYWORDS = (
    os.environ.get("VISION_AGENT_FATIGUE_REQUIRE_KEYWORDS", "1").strip().lower()
    not in {"0", "false", "no"}
)
AGENT_STATE_PATH = os.environ.get(
    "COGNIGUARD_AGENT_STATE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_state.json"),
)
SESSION_LOG_PATH = os.environ.get(
    "COGNIGUARD_SESSION_LOG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions", "session.jsonl"),
)


class WebcamTrack(VideoStreamTrack):
    """Minimal aiortc track that receives frames pushed from OpenCV."""

    def __init__(self) -> None:
        super().__init__()
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._time_base = Fraction(1, 90000)

    def push_frame(self, bgr_frame) -> None:
        frame_copy = bgr_frame.copy()
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(frame_copy)

    async def recv(self) -> av.VideoFrame:
        bgr = await self._queue.get()
        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
        frame.pts = int(time.monotonic() * 90000.0)
        frame.time_base = self._time_base
        return frame


def _parse_decision(text: str):
    raw = (text or "").strip()
    data = {}

    json_candidates = []
    for m in re.finditer(
        r"```(?:json)?\s*(\{[\s\S]*?\})\s*```",
        raw,
        flags=re.IGNORECASE,
    ):
        json_candidates.append(m.group(1).strip())
    if raw.startswith("{"):
        json_candidates.append(raw)
    for m in re.finditer(r"\{[\s\S]*?\}", raw):
        json_candidates.append(m.group(0).strip())

    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                data = parsed
                break
        except Exception:
            continue

    state = str(data.get("state", "")).strip().title()
    if state not in {"Focused", "Fatigued", "Distracted"}:
        raw_l = raw.lower()
        if "fatigued" in raw_l or "fatigue" in raw_l:
            state = "Fatigued"
        elif "focused" in raw_l:
            state = "Focused"
        else:
            state = "Distracted"

    conf_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', raw, flags=re.IGNORECASE)
    try:
        confidence = float(data.get("confidence", conf_match.group(1) if conf_match else 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = str(data.get("reason", "")).strip()
    if not reason:
        reason_match = re.search(
            r'"reason"\s*:\s*"([^"]+)"',
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if reason_match:
            reason = reason_match.group(1).strip()
    if not reason:
        reason = raw[:220] if raw else ""

    reason = re.sub(r"```(?:json)?|```", "", reason, flags=re.IGNORECASE)
    reason = re.sub(r"\s+", " ", reason).strip(" \t\r\n\"'")
    if (
        not reason
        or reason.lower().startswith("json")
        or reason.startswith("{")
    ):
        if state == "Focused":
            reason = "User appears attentive and visually engaged."
        elif state == "Fatigued":
            reason = "Visible fatigue cues detected from face and posture."
        else:
            reason = "Attention appears off-screen or inconsistent."
    return {"state": state, "confidence": confidence, "reason": reason}


def _wrap_text(text, max_chars=52):
    words = str(text).split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _wrap_text_pixels(text, max_width, font, scale, thickness, max_lines=2):
    words = str(text or "").split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        w, _ = cv2.getTextSize(candidate, font, scale, thickness)[0]
        if w <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
            if len(lines) >= max_lines - 1:
                break
    lines.append(current)
    lines = lines[:max_lines]
    if len(words) > 0 and len(lines) == max_lines:
        # add ellipsis if there are remaining words
        joined = " ".join(lines)
        if len(joined.split()) < len(words):
            tail = lines[-1]
            while tail:
                test = f"{tail}..."
                w, _ = cv2.getTextSize(test, font, scale, thickness)[0]
                if w <= max_width:
                    lines[-1] = test
                    break
                tail = tail[:-1]
            if not tail:
                lines[-1] = "..."
    return lines


def _rect_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, float(ix2 - ix1))
    ih = max(0.0, float(iy2 - iy1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1.0, float(ax2 - ax1) * float(ay2 - ay1))
    area_b = max(1.0, float(bx2 - bx1) * float(by2 - by1))
    return float(inter / max(1.0, area_a + area_b - inter))


def _count_eyes_in_face(gray, face_rect, eye_cascade):
    if eye_cascade is None:
        return 0
    fx, fy, fw, fh = face_rect
    y_top = fy
    y_bottom = min(gray.shape[0], fy + int(fh * 0.62))
    x_left = max(0, fx)
    x_right = min(gray.shape[1], fx + fw)
    if y_bottom <= y_top or x_right <= x_left:
        return 0
    roi = gray[y_top:y_bottom, x_left:x_right]
    min_eye = max(8, int(min(fw, fh) * 0.10))
    eyes = eye_cascade.detectMultiScale(
        roi,
        scaleFactor=1.10,
        minNeighbors=3,
        minSize=(min_eye, min_eye),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return int(len(eyes))


def _detect_primary_face_bbox(
    frame,
    frontal_cascade,
    profile_cascade=None,
    preferred_bbox=None,
    eye_cascade=None,
):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    h, w = gray.shape[:2]
    frame_area = float(h * w)
    min_side = max(48, int(min(h, w) * FACE_MIN_SIZE_RATIO))
    min_size = (min_side, min_side)

    candidates = []
    frontal_passes = (
        (1.08, max(3, FACE_FRONTAL_MIN_NEIGHBORS)),
        (1.05, max(2, FACE_FRONTAL_MIN_NEIGHBORS - 1)),
    )
    for scale_factor, min_neighbors in frontal_passes:
        faces = frontal_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for (fx, fy, fw, fh) in faces:
            candidates.append((int(fx), int(fy), int(fw), int(fh), "frontal"))
        if candidates:
            break

    if profile_cascade is not None:
        for img, flipped in ((gray, False), (cv2.flip(gray, 1), True)):
            faces_p = profile_cascade.detectMultiScale(
                img,
                scaleFactor=1.10,
                minNeighbors=3,
                minSize=min_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            for (fx, fy, fw, fh) in faces_p:
                x = int(w - (fx + fw)) if flipped else int(fx)
                candidates.append((x, int(fy), int(fw), int(fh), "profile"))

    if not candidates:
        return None, 0.0

    diag = max(1.0, float((w * w + h * h) ** 0.5))
    preferred_center = None
    if preferred_bbox is not None:
        px1, py1, px2, py2 = preferred_bbox
        preferred_center = ((px1 + px2) * 0.5, (py1 + py2) * 0.5)

    best = None
    best_score = float("-inf")
    for cand in candidates:
        fx, fy, fw, fh, source = cand
        if fw <= 0 or fh <= 0:
            continue
        aspect = float(fw) / max(1.0, float(fh))
        if aspect < FACE_ASPECT_MIN or aspect > FACE_ASPECT_MAX:
            continue
        center_y_ratio = (fy + (fh * 0.5)) / max(1.0, float(h))
        if center_y_ratio > FACE_CENTER_MAX_Y_RATIO:
            continue

        area_ratio = float((fw * fh) / max(1.0, frame_area))
        score = area_ratio
        eye_count = _count_eyes_in_face(gray, (fx, fy, fw, fh), eye_cascade)
        if source == "frontal":
            score += min(2, eye_count) * 0.15
            if FACE_REQUIRE_EYES and eye_count == 0:
                score -= 0.75

        if preferred_center is not None and preferred_bbox is not None:
            cx = fx + (fw * 0.5)
            cy = fy + (fh * 0.5)
            dx = float(cx - preferred_center[0])
            dy = float(cy - preferred_center[1])
            dist_norm = min(1.0, ((dx * dx + dy * dy) ** 0.5) / diag)
            cand_bbox = (fx, fy, fx + fw, fy + fh)
            iou = _rect_iou(cand_bbox, preferred_bbox)
            score += (0.55 * iou) - (0.30 * dist_norm)
            if FACE_REQUIRE_EYES and source == "frontal" and eye_count == 0 and iou > 0.35:
                score += 0.20
        if score > best_score:
            best_score = score
            best = (fx, fy, fw, fh)

    if best is None:
        return None, 0.0
    area_ratio = (best[2] * best[3]) / max(1.0, frame_area)
    if area_ratio < MIN_FACE_AREA_RATIO:
        return None, area_ratio
    return best, area_ratio


def _smooth_bbox(prev_bbox, next_bbox):
    if prev_bbox is None:
        return next_bbox

    def _step(prev, target, smooth, deadzone, max_step):
        delta = float(target) - float(prev)
        if abs(delta) <= deadzone:
            return float(prev)
        step = delta * (1.0 - smooth)
        if step > max_step:
            step = max_step
        elif step < -max_step:
            step = -max_step
        return float(prev) + step

    px1, py1, px2, py2 = [float(v) for v in prev_bbox]
    nx1, ny1, nx2, ny2 = [float(v) for v in next_bbox]

    pcx = (px1 + px2) * 0.5
    pcy = (py1 + py2) * 0.5
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)

    ncx = (nx1 + nx2) * 0.5
    ncy = (ny1 + ny2) * 0.5
    nw = max(1.0, nx2 - nx1)
    nh = max(1.0, ny2 - ny1)

    cx = _step(
        pcx,
        ncx,
        smooth=TRACKER_POS_SMOOTH_FACTOR,
        deadzone=TRACKER_DEADZONE_PX,
        max_step=TRACKER_MAX_STEP_PX,
    )
    cy = _step(
        pcy,
        ncy,
        smooth=TRACKER_POS_SMOOTH_FACTOR,
        deadzone=TRACKER_DEADZONE_PX,
        max_step=TRACKER_MAX_STEP_PX,
    )
    w = _step(
        pw,
        nw,
        smooth=TRACKER_SIZE_SMOOTH_FACTOR,
        deadzone=TRACKER_DEADZONE_PX * 2.0,
        max_step=TRACKER_MAX_STEP_PX * 1.3,
    )
    h = _step(
        ph,
        nh,
        smooth=TRACKER_SIZE_SMOOTH_FACTOR,
        deadzone=TRACKER_DEADZONE_PX * 2.0,
        max_step=TRACKER_MAX_STEP_PX * 1.3,
    )
    w = max(12.0, w)
    h = max(12.0, h)
    return (
        int(round(cx - (w * 0.5))),
        int(round(cy - (h * 0.5))),
        int(round(cx + (w * 0.5))),
        int(round(cy + (h * 0.5))),
    )


def _draw_cinematic_face_corners(frame, bbox):
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return
    w = x2 - x1
    h = y2 - y1
    corner = max(18, min(48, int(min(w, h) * 0.25)))
    c = TRACKER_COLOR
    t = TRACKER_THICKNESS

    # Top-left
    cv2.ellipse(frame, (x1 + corner, y1 + corner), (corner, corner), 0, 180, 270, c, t, cv2.LINE_AA)
    # Top-right
    cv2.ellipse(frame, (x2 - corner, y1 + corner), (corner, corner), 0, 270, 360, c, t, cv2.LINE_AA)
    # Bottom-left
    cv2.ellipse(frame, (x1 + corner, y2 - corner), (corner, corner), 0, 90, 180, c, t, cv2.LINE_AA)
    # Bottom-right
    cv2.ellipse(frame, (x2 - corner, y2 - corner), (corner, corner), 0, 0, 90, c, t, cv2.LINE_AA)


def _draw_dashboard(frame, decision, latency_ms, has_face, inference_fps):
    h, w = frame.shape[:2]
    state = str(decision.get("state", "Analyzing")).strip() or "Analyzing"
    reason = str(decision.get("reason", "Collecting first decision...")).strip()
    latency_s = max(0.0, float(latency_ms)) / 1000.0
    fps_txt = f"{inference_fps:.1f}" if float(inference_fps) > 0.0 else "N/A"

    # NVIDIA-like top strip
    strip_x = 10
    strip_y = 8
    strip_h = 24
    strip_w = min(w - 20, 620)
    overlay = frame.copy()
    cv2.rectangle(overlay, (strip_x, strip_y), (strip_x + strip_w, strip_y + strip_h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    cv2.rectangle(frame, (strip_x, strip_y), (strip_x + strip_w, strip_y + strip_h), (160, 160, 160), 1)
    strip_text = f"FPS {fps_txt} | LAT {latency_s:.1f}s | FACE {'True' if has_face else 'False'} | STATE {state.upper()}"
    cv2.putText(frame, strip_text, (strip_x + 8, strip_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (235, 235, 235), 1, cv2.LINE_AA)


def _write_agent_state(decision, latency_ms, fps, latency_avg_ms, append_log=True):
    timestamp_epoch = float(time.time())
    timestamp_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "state": str(decision.get("state", "Unknown")),
        "confidence": float(decision.get("confidence", 0.0)),
        "reason": str(decision.get("reason", "No reason")),
        "latency_ms": int(round(float(latency_ms))),
        "latency_avg_ms": int(round(float(latency_avg_ms))),
        "fps": round(float(max(0.0, fps)), 2),
        "timestamp": timestamp_iso,
        "timestamp_epoch": timestamp_epoch,
        "model": MODEL_NAME,
        "vision_agent": True,
    }
    os.makedirs(os.path.dirname(AGENT_STATE_PATH) or ".", exist_ok=True)
    temp_path = f"{AGENT_STATE_PATH}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(temp_path, AGENT_STATE_PATH)

    if append_log:
        os.makedirs(os.path.dirname(SESSION_LOG_PATH) or ".", exist_ok=True)
        with open(SESSION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")


async def _ask_agent(session: TestSession):
    conversation = getattr(session, "_conversation", None)
    if conversation is not None and hasattr(conversation, "messages"):
        if len(conversation.messages) > MAX_CONVERSATION_MESSAGES:
            conversation.messages = conversation.messages[-MAX_CONVERSATION_MESSAGES:]

    # Include a changing token so the model re-evaluates current frames each round.
    prompt = (
        "Analyze only the latest camera frames. "
        "Ignore previous responses. "
        f"tick={time.time():.3f}. "
        "Return strict JSON only: "
        "{\"state\":\"Focused|Fatigued|Distracted\",\"confidence\":0.0-1.0,"
        "\"reason\":\"short\"}."
    )
    start = time.perf_counter()
    response = await session.llm.simple_response(prompt)
    latency_ms = (time.perf_counter() - start) * 1000.0
    decision = _parse_decision(response.text or "")
    return decision, float(latency_ms)


def _is_non_visual_result(decision):
    reason = str(decision.get("reason", "")).strip().lower()
    if not reason:
        return True
    blocked_patterns = (
        "no camera access",
        "can't access your camera feed",
        "cannot access your camera feed",
        "no visual data",
        "visual data provided",
        "please take a screenshot",
        "share a recent frame",
    )
    return any(pattern in reason for pattern in blocked_patterns)


def _has_fatigue_keywords(reason):
    r = str(reason or "").strip().lower()
    if not r:
        return False
    keys = (
        "eyes closed",
        "eye closure",
        "prolonged blink",
        "drooping eyelid",
        "droopy eyelid",
        "yawn",
        "head nod",
        "nodding",
        "drowsy",
        "sleepy",
    )
    return any(k in r for k in keys)


def _apply_fatigue_guard(
    decision,
    fatigue_streak,
    non_fatigue_streak,
    fatigue_locked,
):
    state = str(decision.get("state", "Distracted")).strip().title()
    confidence = float(max(0.0, min(1.0, decision.get("confidence", 0.0))))
    reason = str(decision.get("reason", "")).strip()

    fatigue_candidate = state == "Fatigued" and confidence >= FATIGUE_MIN_CONFIDENCE
    if fatigue_candidate and FATIGUE_REQUIRE_REASON_KEYWORDS:
        fatigue_candidate = _has_fatigue_keywords(reason)

    if fatigue_candidate:
        fatigue_streak += 1
        non_fatigue_streak = 0
    else:
        non_fatigue_streak += 1
        fatigue_streak = max(0, fatigue_streak - 1)

    if fatigue_locked:
        if non_fatigue_streak >= max(1, FATIGUE_RECOVER_STREAK):
            fatigue_locked = False
    elif fatigue_streak >= max(1, FATIGUE_CONFIRM_STREAK):
        fatigue_locked = True

    adjusted = dict(decision)
    if fatigue_locked:
        adjusted["state"] = "Fatigued"
        adjusted["confidence"] = max(confidence, 0.75)
        if state != "Fatigued":
            adjusted["reason"] = "Fatigue state held until recovery trend is observed."
    elif state == "Fatigued":
        adjusted["state"] = "Distracted"
        adjusted["confidence"] = min(confidence, 0.65)
        if fatigue_streak > 0:
            adjusted["reason"] = "Possible fatigue cues detected; awaiting consistent evidence."
        else:
            adjusted["reason"] = "Resting expression detected; fatigue not confirmed."

    return adjusted, fatigue_streak, non_fatigue_streak, fatigue_locked


def _open_camera():
    backend_options = []
    if CAMERA_BACKEND == "dshow":
        backend_options = [("dshow", cv2.CAP_DSHOW), ("default", None)]
    elif CAMERA_BACKEND == "msmf":
        backend_options = [("msmf", cv2.CAP_MSMF), ("default", None)]
    elif CAMERA_BACKEND == "default":
        backend_options = [("default", None)]
    else:
        backend_options = [
            ("default", None),
            ("msmf", cv2.CAP_MSMF),
            ("dshow", cv2.CAP_DSHOW),
        ]

    for backend_name, backend in backend_options:
        cap = cv2.VideoCapture(CAMERA_INDEX) if backend is None else cv2.VideoCapture(CAMERA_INDEX, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            cap.release()
            continue

        ready = False
        mean_levels = []
        for _ in range(15):
            ok, frame = cap.read()
            if ok and frame is not None:
                ready = True
                mean_levels.append(float(frame.mean()))
            time.sleep(0.03)

        looks_black = bool(mean_levels) and (sum(mean_levels) / len(mean_levels)) < 5.0
        if ready and not looks_black:
            print(f"[camera] Using index={CAMERA_INDEX} backend={backend_name}")
            return cap
        if ready and looks_black:
            print(
                f"[camera] Rejected index={CAMERA_INDEX} backend={backend_name} "
                "(black stream detected)"
            )
        cap.release()

    raise RuntimeError(
        "Could not open a readable webcam stream. "
        "Close other camera apps/processes or set VISION_AGENT_CAMERA_INDEX / "
        "VISION_AGENT_CAMERA_BACKEND (auto|dshow|msmf)."
    )


async def run_session():
    if not API_KEY:
        raise RuntimeError(
            "Missing API key. Set OPENAI_API_KEY (preferred) or "
            "VISION_AGENT_API_KEY / OLLAMA_API_KEY."
        )

    # Keep explicit reference to SDK-level agent symbol.
    _vision_agent_symbol = VisionAgent
    _ = _vision_agent_symbol

    llm = openai.ChatCompletionsVLM(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=API_BASE_URL,
        fps=STREAM_FPS,
        frame_buffer_seconds=2,
        frame_width=640,
        frame_height=480,
    )

    cap = _open_camera()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face_detector_available = not face_cascade.empty()
    if not face_detector_available:
        print("[warn] Face detector unavailable; no-face guard is disabled.")
    profile_face_cascade = None
    if FACE_PROFILE_ENABLED:
        profile_face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        if profile_face_cascade.empty():
            profile_face_cascade = None
            print("[warn] Profile face cascade unavailable; frontal-only mode.")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    if eye_cascade.empty():
        eye_cascade = None
        print("[warn] Eye cascade unavailable; shoulder false-positive filtering reduced.")

    track = WebcamTrack()
    inference_task = None
    last_log_time = 0.0
    failed_reads = 0
    frame_counter = 0
    has_face = True
    no_face_streak = 0
    face_miss_cycles = 0
    latest_face_bbox = None
    smoothed_face_bbox = None
    fatigue_streak = 0
    non_fatigue_streak = 0
    fatigue_locked = False
    last_decision = {
        "state": "Analyzing",
        "confidence": 0.0,
        "reason": "Collecting first decision...",
    }
    last_latency_ms = 0.0
    latency_avg_ms = 0.0
    inference_fps = 0.0
    last_inference_done_ts = None
    last_no_face_write_ts = 0.0

    try:
        async with TestSession(llm=llm, instructions=AGENT_PROMPT) as session:
            await llm.watch_video_track(track)
            # Give the model a short warmup so early responses include video context.
            last_log_time = time.monotonic()

            while True:
                ok, frame = cap.read()
                if not ok:
                    failed_reads += 1
                    if failed_reads >= 45:
                        raise RuntimeError(
                            "Webcam stream read failed repeatedly. "
                            "The camera is likely busy or disconnected."
                        )
                    await asyncio.sleep(0.005)
                    continue
                failed_reads = 0

                frame = cv2.flip(frame, 1)
                track.push_frame(frame)
                frame_counter += 1

                if face_detector_available and (frame_counter % max(1, FACE_DETECT_EVERY_N) == 0):
                    preferred_bbox = smoothed_face_bbox or latest_face_bbox
                    largest_face, _ = _detect_primary_face_bbox(
                        frame,
                        face_cascade,
                        profile_face_cascade,
                        preferred_bbox=preferred_bbox,
                        eye_cascade=eye_cascade,
                    )
                    frame_h, frame_w = frame.shape[:2]
                    if largest_face is not None:
                        face_miss_cycles = 0
                        has_face = True
                        fx, fy, fw, fh = largest_face
                        pad_x = int(fw * 0.18)
                        pad_y = int(fh * 0.22)
                        x1 = max(0, fx - pad_x)
                        y1 = max(0, fy - pad_y)
                        x2 = min(frame_w - 1, fx + fw + pad_x)
                        y2 = min(frame_h - 1, fy + fh + pad_y)
                        latest_face_bbox = (x1, y1, x2, y2)
                    else:
                        face_miss_cycles += 1
                        has_face = (
                            latest_face_bbox is not None
                            and face_miss_cycles <= max(0, FACE_MISS_CYCLES_TOLERANCE)
                        )
                        if not has_face:
                            latest_face_bbox = None

                if has_face:
                    no_face_streak = 0
                else:
                    no_face_streak += 1

                no_face_mode = no_face_streak >= max(1, NO_FACE_CONSEC_FRAMES)
                if no_face_mode:
                    last_decision = {
                        "state": "Distracted",
                        "confidence": 0.0,
                        "reason": "No face detected in frame.",
                    }
                    last_latency_ms = 0.0
                    latency_avg_ms = latency_avg_ms if latency_avg_ms > 0.0 else 0.0
                    now_wall = time.time()
                    if (now_wall - last_no_face_write_ts) >= 1.0:
                        _write_agent_state(
                            last_decision,
                            0.0,
                            fps=inference_fps,
                            latency_avg_ms=latency_avg_ms,
                            append_log=False,
                        )
                        last_no_face_write_ts = now_wall

                if latest_face_bbox is not None and not no_face_mode:
                    smoothed_face_bbox = _smooth_bbox(smoothed_face_bbox, latest_face_bbox)
                elif no_face_mode:
                    smoothed_face_bbox = None

                now = time.monotonic()
                if (
                    not no_face_mode
                    and
                    inference_task is None
                    and (now - last_log_time) >= INFERENCE_INTERVAL_SECONDS
                ):
                    inference_task = asyncio.create_task(_ask_agent(session))
                    last_log_time = now

                if inference_task is not None and inference_task.done():
                    try:
                        decision, latency_ms = inference_task.result()
                        if no_face_mode:
                            print(
                                f"[{time.strftime('%H:%M:%S')}] "
                                "state=Distracted confidence=0.00 latency=0ms "
                                "reason=No face detected in frame."
                            )
                        elif _is_non_visual_result(decision):
                            print(
                                f"[{time.strftime('%H:%M:%S')}] "
                                "skipped_non_visual_response="
                                f"{decision.get('reason', 'No reason')}"
                            )
                        else:
                            (
                                last_decision,
                                fatigue_streak,
                                non_fatigue_streak,
                                fatigue_locked,
                            ) = _apply_fatigue_guard(
                                decision,
                                fatigue_streak,
                                non_fatigue_streak,
                                fatigue_locked,
                            )
                            last_latency_ms = latency_ms
                            latency_avg_ms = (
                                latency_ms
                                if latency_avg_ms <= 0.0
                                else (0.85 * latency_avg_ms) + (0.15 * latency_ms)
                            )
                            now_done = time.monotonic()
                            if last_inference_done_ts is not None:
                                dt_done = max(1e-3, now_done - last_inference_done_ts)
                                instant_fps = 1.0 / dt_done
                                inference_fps = (
                                    instant_fps
                                    if inference_fps <= 0.0
                                    else (0.80 * inference_fps) + (0.20 * instant_fps)
                                )
                            last_inference_done_ts = now_done
                            _write_agent_state(
                                last_decision,
                                latency_ms,
                                fps=inference_fps,
                                latency_avg_ms=latency_avg_ms,
                            )
                            warn = " [SLOW]" if latency_ms > 1000.0 else ""
                            print(
                                f"[{time.strftime('%H:%M:%S')}] "
                                f"state={last_decision['state']} "
                                f"confidence={last_decision['confidence']:.2f} "
                                f"latency={latency_ms:.0f}ms avg={latency_avg_ms:.0f}ms "
                                f"fps={inference_fps:.2f}{warn} "
                                f"reason={last_decision['reason']}"
                            )
                    except Exception as exc:
                        print(f"[{time.strftime('%H:%M:%S')}] agent_error={type(exc).__name__}: {exc}")
                    finally:
                        inference_task = None

                _draw_dashboard(
                    frame,
                    decision=last_decision,
                    latency_ms=latency_avg_ms if latency_avg_ms > 0.0 else last_latency_ms,
                    has_face=has_face,
                    inference_fps=inference_fps,
                )
                if smoothed_face_bbox is not None:
                    _draw_cinematic_face_corners(frame, smoothed_face_bbox)
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break

                await asyncio.sleep(0.001)
    finally:
        if inference_task is not None and not inference_task.done():
            inference_task.cancel()
            with contextlib.suppress(Exception):
                await inference_task
        cap.release()
        cv2.destroyAllWindows()
        await llm.close()


if __name__ == "__main__":
    asyncio.run(run_session())
