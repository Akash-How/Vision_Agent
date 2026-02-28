import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import json
import os
from datetime import datetime
from collections import deque



LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
LEFT_EYE_IDX = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_IDX = (362, 385, 387, 263, 373, 380)
NOSE_TIP_IDX = 1
BLINK_THRESHOLD = 0.18
BLINK_CONSEC_FRAMES = 3
BLINK_CLOSE_RATIO = 0.72
BLINK_OPEN_RATIO = 0.80
BLINK_MIN_THRESHOLD = 0.14
BLINK_REFRACTORY_FRAMES = 5
DROWSY_CLOSED_FRAMES = 28
GAZE_DEVIATION_THRESHOLD = 0.22
GAZE_WINDOW_SIZE = 120
ATTENTION_SMOOTH_FACTOR = 0.85
EAR_SMOOTH_FACTOR = 0.70
EAR_BASELINE_SMOOTH = 0.97
POSTURE_SMOOTH_FACTOR = 0.90
POSTURE_TILT_SCALE = 5.5
POSTURE_EXCELLENT_THRESHOLD = 0.78
POSTURE_GOOD_THRESHOLD = 0.55
STATE_FOCUSED_ATTENTION_THRESHOLD = 62
FACE_LOCK_LANDMARKS = (10, 67, 297, 33, 263, 61, 291, 199, 152, 234, 454, 168)
FACE_LOCK_DURATION_S = 2.0
FACE_LOCK_BOX_COLOR = (100, 200, 255)
FACE_LOCK_LINE_THICKNESS = 3
FACE_LOCK_CORNER_LENGTH = 40
FACE_LOCK_TRANSITION_START = 0.45
FACE_LOCK_TRANSITION_MIN_REVEAL = 0.25
FACE_LOCK_TRANSITION_MIN_ALPHA = 0.15
FACE_SMOOTH_FACTOR = 0.85
SIGNAL_WINDOW_SECONDS = 30
DROWSY_EVENT_COOLDOWN_SECONDS = 8.0

WINDOW_NAME = "CogniGuard Vision Core"
COGNIGUARD_CAMERA_INDEX = int(os.environ.get("COGNIGUARD_CAMERA_INDEX", "0"))
COGNIGUARD_CAMERA_BACKEND = os.environ.get("COGNIGUARD_CAMERA_BACKEND", "auto").strip().lower()
METRICS_BRIDGE_PATH = os.environ.get(
    "COGNIGUARD_METRICS_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cogniguard_metrics.json"),
)
METRICS_BRIDGE_WRITE_INTERVAL_SECONDS = 0.5
AGENT_STATE_PATH = os.environ.get(
    "COGNIGUARD_AGENT_STATE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_state.json"),
)
AGENT_STATE_STALE_SECONDS = 10.0

_TOOLS_METRICS_LOCK = threading.Lock()
_TOOLS_METRICS = {
    "attention": 0.0,
    "gaze_off": 0.0,
    "blink_rate": 0.0,
    "posture_score": 0.0,
    "drowsy_events": 0,
    "fatigue_detected": False,
    "state": "Distracted",
    "confidence": 0.0,
    "ai_state": "Analyzing",
    "reason": "Collecting snapshot for AI reasoning...",
    "secondary_reason": "",
    "risk": "Medium",
    "snapshot": "",
    "updated_at": 0.0,
}
def classify_cognitive_state(attention_percent, drowsy):
    if bool(drowsy):
        return "Fatigued"
    if float(attention_percent) > STATE_FOCUSED_ATTENTION_THRESHOLD:
        return "Focused"
    return "Distracted"


def estimate_state_confidence(attention_percent, posture_score, drowsy):
    attention_norm = float(np.clip(float(attention_percent) / 100.0, 0.0, 1.0))
    posture_norm = float(np.clip(float(posture_score), 0.0, 1.0))
    if drowsy:
        return float(np.clip(0.65 + 0.35 * attention_norm, 0.0, 1.0))
    return float(np.clip(0.55 * attention_norm + 0.45 * posture_norm, 0.0, 1.0))


def publish_tools_metrics(payload):
    with _TOOLS_METRICS_LOCK:
        _TOOLS_METRICS.update(payload)
        _TOOLS_METRICS["updated_at"] = time.time()


def get_tools_metrics():
    with _TOOLS_METRICS_LOCK:
        return dict(_TOOLS_METRICS)


def compute_risk_label(attention_percent, posture_score, drowsy):
    if bool(drowsy):
        return "High"
    if float(attention_percent) < 45.0 or float(posture_score) < 0.45:
        return "High"
    if float(attention_percent) < 70.0 or float(posture_score) < 0.62:
        return "Medium"
    return "Low"


def write_metrics_bridge(payload, path=METRICS_BRIDGE_PATH):
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(temp_path, path)


def _parse_agent_timestamp_to_epoch(value):
    if value is None:
        return 0.0
    try:
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return 0.0
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return 0.0


def get_vision_agent_state():
    fallback = {
        "state": "Unknown",
        "confidence": 0.0,
        "reason": "Waiting for Vision Agent",
        "latency_ms": 0.0,
        "latency_avg_ms": 0.0,
        "fps": 0.0,
        "timestamp": "",
        "timestamp_epoch": 0.0,
        "model": "Unknown",
        "status": "Waiting",
        "vision_agent": False,
    }

    try:
        if not os.path.exists(AGENT_STATE_PATH):
            return fallback

        state = dict(fallback)
        with open(AGENT_STATE_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            state.update(loaded)

        state["state"] = str(state.get("state", "Unknown")).strip() or "Unknown"
        state["reason"] = str(state.get("reason", fallback["reason"])).strip() or fallback["reason"]
        state["model"] = str(state.get("model", fallback["model"])).strip() or fallback["model"]
        state["confidence"] = float(np.clip(float(state.get("confidence", 0.0)), 0.0, 1.0))
        state["latency_ms"] = float(max(0.0, float(state.get("latency_ms", 0.0))))
        state["latency_avg_ms"] = float(max(0.0, float(state.get("latency_avg_ms", state["latency_ms"]))))
        state["fps"] = float(max(0.0, float(state.get("fps", 0.0))))
        state["timestamp"] = str(state.get("timestamp", "")).strip()
        state["timestamp_epoch"] = float(
            state.get("timestamp_epoch", _parse_agent_timestamp_to_epoch(state["timestamp"]))
        )
        state["vision_agent"] = bool(state.get("vision_agent", False))

        is_fresh = (time.time() - state["timestamp_epoch"]) <= AGENT_STATE_STALE_SECONDS
        if not (state["vision_agent"] and is_fresh):
            state["vision_agent"] = False
            state["status"] = "Stale"
            if state["timestamp_epoch"] > 0:
                state["reason"] = "Vision Agent state is stale."
            else:
                state["reason"] = fallback["reason"]
        else:
            state["status"] = "Analyzing"
        return state
    except Exception:
        return fallback


def compute_posture_score(pose_landmarks):
    """Return posture score in [0, 1] based on shoulder level difference."""
    if pose_landmarks is None:
        return 0.0

    left_shoulder = pose_landmarks.landmark[LEFT_SHOULDER_IDX]
    right_shoulder = pose_landmarks.landmark[RIGHT_SHOULDER_IDX]
    visibility = min(float(left_shoulder.visibility), float(right_shoulder.visibility))
    if visibility < 0.25:
        return 0.0

    tilt = float(np.abs(left_shoulder.y - right_shoulder.y))
    posture_score = max(0.0, 1.0 - tilt * POSTURE_TILT_SCALE)
    posture_score *= 0.85 + 0.15 * visibility
    return float(posture_score)


def landmark_xy(landmarks, idx):
    lm = landmarks.landmark[idx]
    return np.array([lm.x, lm.y], dtype=np.float32)


def compute_eye_aspect_ratio(landmarks, eye_indices):
    p1 = landmark_xy(landmarks, eye_indices[0])
    p2 = landmark_xy(landmarks, eye_indices[1])
    p3 = landmark_xy(landmarks, eye_indices[2])
    p4 = landmark_xy(landmarks, eye_indices[3])
    p5 = landmark_xy(landmarks, eye_indices[4])
    p6 = landmark_xy(landmarks, eye_indices[5])

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal <= 1e-6:
        return 0.0
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def compute_ear(face_landmarks):
    left_ear = compute_eye_aspect_ratio(face_landmarks, LEFT_EYE_IDX)
    right_ear = compute_eye_aspect_ratio(face_landmarks, RIGHT_EYE_IDX)
    return float((left_ear + right_ear) * 0.5)


def detect_looking_at_screen(face_landmarks, frame_shape):
    h, w = frame_shape[:2]
    x1, _, x2, _ = compute_face_bbox(face_landmarks, frame_shape)
    face_width = max(1.0, float(x2 - x1))
    nose_x = float(face_landmarks.landmark[NOSE_TIP_IDX].x) * float(w)
    face_center_x = 0.5 * float(x1 + x2)
    deviation = abs(nose_x - face_center_x) / face_width
    looking_at_screen = deviation < GAZE_DEVIATION_THRESHOLD
    return bool(looking_at_screen), float(deviation)


def smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def exp_smooth(prev_value, current_value, smooth_factor=FACE_SMOOTH_FACTOR):
    if prev_value is None:
        return current_value.astype(np.float32, copy=True)
    return (prev_value * smooth_factor) + (current_value * (1.0 - smooth_factor))


def exp_smooth_scalar(prev_value, current_value, smooth_factor):
    if prev_value is None:
        return float(current_value)
    return float(prev_value * smooth_factor + current_value * (1.0 - smooth_factor))


class RollingSignalAggregator:
    """Maintains a rolling cognitive snapshot over a fixed time window."""

    def __init__(self, window_seconds=SIGNAL_WINDOW_SECONDS):
        self.window_seconds = float(window_seconds)
        self.ear_samples = deque()
        self.posture_samples = deque()
        self.gaze_samples = deque()
        self.drowsy_samples = deque()
        self.blink_timestamps = deque()
        self.drowsy_event_timestamps = deque()
        self.prev_drowsy = False
        self.last_drowsy_event_ts = -1e9

    def _prune(self, now_ts):
        cutoff = now_ts - self.window_seconds
        while self.ear_samples and self.ear_samples[0][0] < cutoff:
            self.ear_samples.popleft()
        while self.posture_samples and self.posture_samples[0][0] < cutoff:
            self.posture_samples.popleft()
        while self.gaze_samples and self.gaze_samples[0][0] < cutoff:
            self.gaze_samples.popleft()
        while self.drowsy_samples and self.drowsy_samples[0][0] < cutoff:
            self.drowsy_samples.popleft()
        while self.blink_timestamps and self.blink_timestamps[0] < cutoff:
            self.blink_timestamps.popleft()
        while self.drowsy_event_timestamps and self.drowsy_event_timestamps[0] < cutoff:
            self.drowsy_event_timestamps.popleft()

    def update(self, now_ts, ear, posture, looking_at_screen, drowsy, blink_increment):
        self.ear_samples.append((now_ts, float(ear)))
        self.posture_samples.append((now_ts, float(posture)))
        self.gaze_samples.append((now_ts, bool(looking_at_screen)))
        self.drowsy_samples.append((now_ts, bool(drowsy)))
        for _ in range(max(0, int(blink_increment))):
            self.blink_timestamps.append(now_ts)

        # Count a drowsy event only on False->True transitions with cooldown.
        if drowsy and (not self.prev_drowsy):
            if (now_ts - self.last_drowsy_event_ts) >= DROWSY_EVENT_COOLDOWN_SECONDS:
                self.drowsy_event_timestamps.append(now_ts)
                self.last_drowsy_event_ts = now_ts
        self.prev_drowsy = bool(drowsy)
        self._prune(now_ts)

    def snapshot(self, now_ts):
        self._prune(now_ts)

        if self.posture_samples:
            posture_avg = float(np.mean([v for _, v in self.posture_samples]))
        else:
            posture_avg = 0.0

        if self.gaze_samples:
            gaze_off_percent = 100.0 * (
                sum(1 for _, looking in self.gaze_samples if not looking) / len(self.gaze_samples)
            )
        else:
            gaze_off_percent = 100.0

        drowsy_events = int(len(self.drowsy_event_timestamps))
        blink_rate = int(round(len(self.blink_timestamps) * (60.0 / self.window_seconds)))

        return {
            "blink_rate": blink_rate,
            "gaze_off_percent": float(gaze_off_percent),
            "posture_avg": float(posture_avg),
            "drowsy_events": drowsy_events,
        }


def format_cognitive_snapshot(snapshot):
    blink_rate = float(snapshot.get("blink_rate", 0.0))
    gaze_off = float(snapshot.get("gaze_off_percent", 0.0))
    posture_avg = float(snapshot.get("posture_avg", 0.0))
    drowsy_events = int(snapshot.get("drowsy_events", 0))
    return (
        f"BlinkRate={blink_rate:.1f}, GazeOff={gaze_off:.1f}, "
        f"PostureAvg={posture_avg:.2f}, DrowsyEvents={drowsy_events}"
    )


def parse_ai_reasoning(text):
    content = (text or "").strip()
    if not content:
        return "Unknown", "No response from model."

    state = "Unknown"
    reason = content
    if "State:" in content and "Reason:" in content:
        state_part = content.split("State:", 1)[1].split("Reason:", 1)[0].strip()
        reason_part = content.split("Reason:", 1)[1].strip()
        if state_part:
            state = state_part
        if reason_part:
            reason = reason_part
    elif content.lower().startswith("state:"):
        state = content.split(":", 1)[1].strip()
        reason = "No reason provided."

    return state, reason


def fallback_reasoning_from_snapshot(snapshot):
    blink_rate = float(snapshot.get("blink_rate", 0.0))
    gaze_off = float(snapshot.get("gaze_off_percent", 0.0))
    posture_avg = float(snapshot.get("posture_avg", 0.0))
    drowsy_events = int(snapshot.get("drowsy_events", 0))

    if drowsy_events >= 1 or blink_rate >= 30:
        state = "Fatigued"
        reason = "Frequent drowsy signals or elevated blink rate in the recent window."
    elif gaze_off >= 55 and posture_avg < 0.45:
        state = "BurnedOut"
        reason = "Sustained high gaze-off with poor posture suggests cognitive overload."
    elif gaze_off >= 35 or posture_avg < 0.55:
        state = "Drifting"
        reason = "Attention drift or posture instability detected over the rolling snapshot."
    else:
        state = "Focused"
        reason = "Stable gaze and posture with low fatigue indicators."

    return state, reason


def call_ollama_reasoning(snapshot):
    if requests is None:
        raise RuntimeError("requests library is not available.")

    blink_rate = float(snapshot.get("blink_rate", 0.0))
    gaze_off = float(snapshot.get("gaze_off_percent", 0.0))
    posture_avg = float(snapshot.get("posture_avg", 0.0))
    drowsy_events = int(snapshot.get("drowsy_events", 0))
    user_prompt = OLLAMA_USER_PROMPT_TEMPLATE.format(
        blink_rate=f"{blink_rate:.1f}",
        gaze_off=f"{gaze_off:.1f}",
        posture_avg=f"{posture_avg:.2f}",
        drowsy_events=drowsy_events,
    )

    response = requests.post(
        OLLAMA_ENDPOINT,
        headers={"Content-Type": "application/json"},
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": OLLAMA_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 96,
            },
        },
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    msg = data.get("message", {}) or {}
    content = str(msg.get("content", "")).strip()

    if not content:
        state, reason = fallback_reasoning_from_snapshot(snapshot)
        content = f"State: {state} Reason: {reason}"
    else:
        state, reason = parse_ai_reasoning(content)
    return {
        "state": state,
        "reason": reason,
        "raw": content,
        "snapshot": format_cognitive_snapshot(snapshot),
    }


def ai_reasoning_worker(stop_event, snapshot_lock, snapshot_ref, ai_lock, ai_ref):
    while not stop_event.is_set():
        with snapshot_lock:
            snapshot = dict(snapshot_ref)

        try:
            result = call_ollama_reasoning(snapshot)
            with ai_lock:
                ai_ref.update(result)
                ai_ref["last_error"] = ""
        except Exception as exc:
            # Keep last successful response if network/API fails, but expose
            # the error when there is no successful response yet.
            err_name = type(exc).__name__
            err_msg = str(exc).strip() or "No details"
            with ai_lock:
                ai_ref["last_error"] = f"{err_name}: {err_msg[:140]}"
                if not ai_ref.get("raw"):
                    ai_ref["state"] = "Unavailable"
                    ai_ref["reason"] = f"AI request failed ({err_name})."

        stop_event.wait(OLLAMA_INTERVAL_SECONDS)


def get_face_lock_targets(face_landmarks, frame_shape):
    h, w = frame_shape[:2]
    points = []
    for idx in FACE_LOCK_LANDMARKS:
        lm = face_landmarks.landmark[idx]
        points.append((lm.x * w, lm.y * h))
    return np.asarray(points, dtype=np.float32)


def build_start_points(center_xy, count, radius):
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False, dtype=np.float32)
    x = center_xy[0] + np.cos(angles) * radius
    y = center_xy[1] + np.sin(angles) * radius
    return np.stack((x, y), axis=1).astype(np.float32)


def compute_face_bbox(face_landmarks, frame_shape, pad=12):
    h, w = frame_shape[:2]
    xs = np.asarray([lm.x for lm in face_landmarks.landmark], dtype=np.float32) * w
    ys = np.asarray([lm.y for lm in face_landmarks.landmark], dtype=np.float32) * h
    x1 = int(np.clip(np.min(xs) - pad, 0, w - 1))
    y1 = int(np.clip(np.min(ys) - pad, 0, h - 1))
    x2 = int(np.clip(np.max(xs) + pad, 0, w - 1))
    y2 = int(np.clip(np.max(ys) + pad, 0, h - 1))
    return x1, y1, x2, y2


def draw_corner_curve(frame, p0, p1, p2, color, thickness):
    t = np.linspace(0.0, 1.0, 20, dtype=np.float32)[:, None]
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    points = (1.0 - t) ** 2 * p0 + 2.0 * (1.0 - t) * t * p1 + (t**2) * p2
    points = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [points], False, color, thickness, lineType=cv2.LINE_AA)


def draw_cinematic_face_corners(frame, bbox, reveal=1.0, alpha=1.0):
    reveal = float(np.clip(reveal, 0.0, 1.0))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if reveal <= 0.0 or alpha <= 0.0:
        return

    x1, y1, x2, y2 = bbox
    max_len = min(
        FACE_LOCK_CORNER_LENGTH,
        max(8, (x2 - x1) // 2 - 2),
        max(8, (y2 - y1) // 2 - 2),
    )
    l = max(6, int(max_len * reveal))
    target = frame if alpha >= 0.995 else frame.copy()

    draw_corner_curve(
        target,
        (x1 + l, y1),
        (x1, y1),
        (x1, y1 + l),
        FACE_LOCK_BOX_COLOR,
        FACE_LOCK_LINE_THICKNESS,
    )
    draw_corner_curve(
        target,
        (x2 - l, y1),
        (x2, y1),
        (x2, y1 + l),
        FACE_LOCK_BOX_COLOR,
        FACE_LOCK_LINE_THICKNESS,
    )
    draw_corner_curve(
        target,
        (x1, y2 - l),
        (x1, y2),
        (x1 + l, y2),
        FACE_LOCK_BOX_COLOR,
        FACE_LOCK_LINE_THICKNESS,
    )
    draw_corner_curve(
        target,
        (x2 - l, y2),
        (x2, y2),
        (x2, y2 - l),
        FACE_LOCK_BOX_COLOR,
        FACE_LOCK_LINE_THICKNESS,
    )

    if target is not frame:
        cv2.addWeighted(target, alpha, frame, 1.0 - alpha, 0, frame)


def draw_face_lock_visual(
    frame,
    face_landmarks,
    lock_start_time,
    lock_start_points,
    smoothed_bbox,
):
    targets = get_face_lock_targets(face_landmarks, frame.shape)
    if lock_start_time is None or lock_start_points is None:
        center = np.mean(targets, axis=0)
        base_radius = max(40.0, min(frame.shape[0], frame.shape[1]) * 0.22)
        lock_start_points = build_start_points(center, targets.shape[0], base_radius)
        lock_start_time = time.monotonic()

    now = time.monotonic()
    elapsed = now - lock_start_time
    progress = min(1.0, elapsed / FACE_LOCK_DURATION_S)
    bbox = np.asarray(compute_face_bbox(face_landmarks, frame.shape), dtype=np.float32)
    smoothed_bbox = exp_smooth(smoothed_bbox, bbox)

    transition_raw = (progress - FACE_LOCK_TRANSITION_START) / max(
        1e-6, (1.0 - FACE_LOCK_TRANSITION_START)
    )
    transition = smoothstep(np.clip(transition_raw, 0.0, 1.0))
    if transition > 0.0:
        reveal = FACE_LOCK_TRANSITION_MIN_REVEAL + (
            1.0 - FACE_LOCK_TRANSITION_MIN_REVEAL
        ) * transition
        alpha = FACE_LOCK_TRANSITION_MIN_ALPHA + (
            1.0 - FACE_LOCK_TRANSITION_MIN_ALPHA
        ) * transition
        draw_cinematic_face_corners(
            frame,
            tuple(np.round(smoothed_bbox).astype(np.int32).tolist()),
            reveal=reveal,
            alpha=alpha,
        )

    return (
        lock_start_time,
        lock_start_points,
        smoothed_bbox,
    )


def draw_hud(
    frame,
    ai_output,
    posture_score,
    drowsy,
    attention_percent,
    face_present,
):
    """Draw premium Apple-style glass dashboard."""
    panel_x, panel_y = 20, 20
    panel_w, panel_h = 420, 260
    blur_kernel = 45
    opacity = 0.35
    background_tint = (40, 40, 40)
    border_color = (255, 255, 255)
    border_thickness = 1
    corner_radius = 20

    h, w = frame.shape[:2]
    x1 = int(np.clip(panel_x, 0, max(0, w - 1)))
    y1 = int(np.clip(panel_y, 0, max(0, h - 1)))
    x2 = int(np.clip(panel_x + panel_w, x1 + 1, w))
    y2 = int(np.clip(panel_y + panel_h, y1 + 1, h))
    roi_w = x2 - x1
    roi_h = y2 - y1
    if roi_w <= 1 or roi_h <= 1:
        return y1

    roi = frame[y1:y2, x1:x2]

    k = max(3, int(blur_kernel))
    if (k % 2) == 0:
        k += 1
    small_w = max(1, roi_w // 2)
    small_h = max(1, roi_h // 2)
    roi_small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    blur_small = cv2.GaussianBlur(roi_small, (k, k), 0)
    blur_roi = cv2.resize(blur_small, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
    glass = cv2.addWeighted(roi, 1.0 - opacity, blur_roi, opacity, 0)
    tint = np.full_like(glass, background_tint)
    glass = cv2.addWeighted(glass, 0.78, tint, 0.22, 0)

    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    r = int(np.clip(corner_radius, 2, max(2, min(roi_w, roi_h) // 2)))
    cv2.rectangle(mask, (r, 0), (roi_w - r, roi_h), 255, -1)
    cv2.rectangle(mask, (0, r), (roi_w, roi_h - r), 255, -1)
    cv2.circle(mask, (r, r), r, 255, -1)
    cv2.circle(mask, (roi_w - r, r), r, 255, -1)
    cv2.circle(mask, (r, roi_h - r), r, 255, -1)
    cv2.circle(mask, (roi_w - r, roi_h - r), r, 255, -1)

    roi_out = roi.copy()
    roi_out[mask > 0] = glass[mask > 0]
    frame[y1:y2, x1:x2] = roi_out

    border_overlay = frame.copy()
    cv2.line(border_overlay, (x1 + r, y1), (x2 - r, y1), border_color, border_thickness, cv2.LINE_AA)
    cv2.line(border_overlay, (x1 + r, y2 - 1), (x2 - r, y2 - 1), border_color, border_thickness, cv2.LINE_AA)
    cv2.line(border_overlay, (x1, y1 + r), (x1, y2 - r), border_color, border_thickness, cv2.LINE_AA)
    cv2.line(border_overlay, (x2 - 1, y1 + r), (x2 - 1, y2 - r), border_color, border_thickness, cv2.LINE_AA)
    cv2.ellipse(border_overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90, border_color, border_thickness, cv2.LINE_AA)
    cv2.ellipse(border_overlay, (x2 - r - 1, y1 + r), (r, r), 270, 0, 90, border_color, border_thickness, cv2.LINE_AA)
    cv2.ellipse(border_overlay, (x1 + r, y2 - r - 1), (r, r), 90, 0, 90, border_color, border_thickness, cv2.LINE_AA)
    cv2.ellipse(border_overlay, (x2 - r - 1, y2 - r - 1), (r, r), 0, 0, 90, border_color, border_thickness, cv2.LINE_AA)
    cv2.addWeighted(border_overlay, 0.9, frame, 0.1, 0, frame)

    fallback_state = classify_cognitive_state(int(np.clip(attention_percent, 0.0, 100.0)), drowsy)
    state_label = str(ai_output.get("state", fallback_state)).strip() or fallback_state
    confidence_pct = int(round(float(np.clip(float(ai_output.get("confidence", 0.0)), 0.0, 1.0)) * 100.0))
    latency_ms = float(max(0.0, float(ai_output.get("latency_avg_ms", ai_output.get("latency_ms", 0.0)))))
    latency_s = latency_ms / 1000.0
    fps = float(max(0.0, float(ai_output.get("fps", 0.0))))
    reason = str(ai_output.get("reason", "Collecting first decision...")).strip()

    status_colors = {
        "Focused": (60, 200, 120),
        "Distracted": (255, 180, 60),
        "Fatigued": (255, 80, 80),
    }
    state_color = status_colors.get(state_label, (140, 200, 255))
    primary_text = (255, 255, 255)
    secondary_text = (180, 180, 180)
    metric_label = (160, 160, 160)
    metric_value = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_text(text, tx, ty, scale, color, thickness):
        px = x1 + tx
        py = y1 + ty
        cv2.putText(
            frame,
            text,
            (px, py),
            font,
            scale,
            (0, 0, 0),
            thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (px, py),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    # Header
    draw_text("CogniGuard", 40, 60, 0.9, primary_text, 2)
    draw_text("Vision Agent", 40, 85, 0.5, secondary_text, 1)

    # Status badge
    bx1, by1 = x1 + 260, y1 + 60
    bx2, by2 = bx1 + 140, by1 + 40
    badge_overlay = frame.copy()
    cv2.rectangle(badge_overlay, (bx1, by1), (bx2, by2), state_color, -1)
    cv2.addWeighted(badge_overlay, 0.35, frame, 0.65, 0, frame)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
    draw_text(state_label, 280, 87, 0.68, primary_text, 2)

    # Metrics grid
    # Row 1
    draw_text("Confidence", 40, 130, 0.45, metric_label, 1)
    draw_text(f"{confidence_pct}%", 40, 152, 0.70, metric_value, 2)
    draw_text("Latency", 220, 130, 0.45, metric_label, 1)
    draw_text(f"{latency_s:.1f}s", 220, 152, 0.70, metric_value, 2)
    # Row 2
    draw_text("FPS", 40, 180, 0.45, metric_label, 1)
    draw_text(f"{fps:.1f}", 40, 202, 0.70, metric_value, 2)
    draw_text("Face", 220, 180, 0.45, metric_label, 1)
    draw_text("True" if face_present else "False", 220, 202, 0.70, metric_value, 2)

    # Reasoning section
    draw_text("AI Reasoning", 40, 210, 0.45, metric_label, 1)
    reason_lines = wrap_text(reason, max_chars=46)[:2]
    reason_y = 235
    for line in reason_lines:
        draw_text(line, 40, reason_y, 0.52, primary_text, 1)
        reason_y += 20

    return y2


def draw_text_card(
    frame,
    start_x,
    start_y,
    lines,
    width,
    line_spacing=30,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale=0.7,
    color=(255, 255, 255),
    thickness=2,
    box_alpha=0.55,
):
    padding_x = 12
    padding_y = 12
    box_h = (len(lines) * line_spacing) + (padding_y * 2) - 8

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (start_x, start_y),
        (start_x + width, start_y + box_h),
        (16, 16, 16),
        -1,
    )
    cv2.addWeighted(overlay, box_alpha, frame, 1.0 - box_alpha, 0, frame)
    cv2.rectangle(frame, (start_x, start_y), (start_x + width, start_y + box_h), (90, 90, 90), 1)

    for i, text in enumerate(lines):
        y = start_y + padding_y + 18 + i * line_spacing
        # thin shadow for readability on bright backgrounds
        cv2.putText(
            frame,
            text,
            (start_x + padding_x, y),
            font,
            scale,
            (0, 0, 0),
            thickness + 2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (start_x + padding_x, y),
            font,
            scale,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    return start_y + box_h


def draw_signal_debug_overlay(frame, rolling_snapshot, start_y=150):
    start_x = 20
    line_spacing = 30

    blink_rate = float(rolling_snapshot.get("blink_rate", 0.0))
    gaze_off_percent = float(rolling_snapshot.get("gaze_off_percent", 0.0))
    posture_avg = float(rolling_snapshot.get("posture_avg", 0.0))
    drowsy_events = int(rolling_snapshot.get("drowsy_events", 0))

    lines = [
        f"BlinkRate: {blink_rate:.1f}",
        f"GazeOff: {gaze_off_percent:.1f}%",
        f"PostureAvg: {posture_avg:.2f}",
        f"DrowsyEvents: {drowsy_events}",
    ]

    return draw_text_card(
        frame,
        start_x=start_x,
        start_y=start_y,
        lines=lines,
        width=360,
        line_spacing=line_spacing,
        scale=0.68,
        thickness=2,
        box_alpha=0.5,
    )


def wrap_text(text, max_chars=48):
    words = text.split()
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


def draw_ai_reasoning_overlay(frame, ai_output, start_y=300):
    start_x = 20
    line_spacing = 26

    ai_state = str(ai_output.get("state", "Analyzing"))
    reason = str(ai_output.get("reason", "Waiting for first AI response..."))
    last_error = str(ai_output.get("last_error", "")).strip()
    reason_lines = wrap_text(reason, max_chars=50)

    lines = ["AI State:", ai_state, "Reason:"] + reason_lines[:3]
    if last_error:
        lines += ["Error:", last_error[:56]]
    return draw_text_card(
        frame,
        start_x=start_x,
        start_y=start_y,
        lines=lines,
        width=560,
        line_spacing=line_spacing,
        scale=0.66,
        thickness=2,
        box_alpha=0.52,
    )


def _open_cogniguard_camera():
    if COGNIGUARD_CAMERA_BACKEND == "dshow":
        backend_options = [("dshow", cv2.CAP_DSHOW), ("default", None)]
    elif COGNIGUARD_CAMERA_BACKEND == "msmf":
        backend_options = [("msmf", cv2.CAP_MSMF), ("default", None)]
    elif COGNIGUARD_CAMERA_BACKEND == "default":
        backend_options = [("default", None)]
    else:
        # Prefer backend-default first because some virtual cameras present black on DSHOW.
        backend_options = [
            ("default", None),
            ("msmf", cv2.CAP_MSMF),
            ("dshow", cv2.CAP_DSHOW),
        ]

    for backend_name, backend in backend_options:
        cap = (
            cv2.VideoCapture(COGNIGUARD_CAMERA_INDEX)
            if backend is None
            else cv2.VideoCapture(COGNIGUARD_CAMERA_INDEX, backend)
        )
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
            print(
                f"[main-camera] Using index={COGNIGUARD_CAMERA_INDEX} "
                f"backend={backend_name}"
            )
            return cap
        if ready and looks_black:
            print(
                f"[main-camera] Rejected index={COGNIGUARD_CAMERA_INDEX} "
                f"backend={backend_name} (black stream detected)"
            )
        cap.release()

    raise RuntimeError(
        "Could not open webcam stream. Set COGNIGUARD_CAMERA_INDEX and "
        "COGNIGUARD_CAMERA_BACKEND (auto|dshow|msmf)."
    )


def main():
    cap = _open_cogniguard_camera()

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    class DummyLandmarkList:
        def __init__(self, landmarks):
            # Tasks API landmarks have .x, .y, .z, and .visibility depending on task
            # Provide default visibility=1.0 for those missing it.
            for lm in landmarks:
                if not hasattr(lm, "visibility"):
                    lm.visibility = 1.0
            self.landmark = landmarks
    blink_count = 0
    blink_cooldown = 0
    closed_frames = 0
    eyes_closed = False
    ear_filtered = None
    ear_baseline = None
    gaze_off_window = deque(maxlen=GAZE_WINDOW_SIZE)
    attention_percent = 100.0
    posture_smoothed = None
    signal_aggregator = RollingSignalAggregator(window_seconds=SIGNAL_WINDOW_SECONDS)
    rolling_snapshot = {
        "blink_rate": 0,
        "gaze_off_percent": 100.0,
        "posture_avg": 0.0,
        "drowsy_events": 0,
    }

    lock_start_time = None
    lock_start_points = None
    smoothed_bbox = None
    last_bridge_write_ts = 0.0

    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='face_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1)
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='pose_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE)

    with FaceLandmarker.create_from_options(face_options) as face_mesh, \
         PoseLandmarker.create_from_options(pose_options) as pose:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            face_results = face_mesh.detect(mp_image)
            pose_results = pose.detect(mp_image)

            rgb.flags.writeable = True

            face_detected = bool(face_results.face_landmarks)
            
            wrapped_pose = None
            if pose_results and pose_results.pose_landmarks:
                wrapped_pose = DummyLandmarkList(pose_results.pose_landmarks[0])
            posture_raw = compute_posture_score(wrapped_pose)
            posture_smoothed = exp_smooth_scalar(
                posture_smoothed, posture_raw, POSTURE_SMOOTH_FACTOR
            )
            posture_score = posture_smoothed
            ear_value = 0.0
            drowsy = False
            looking_at_screen = False
            blink_increment = 0

            if blink_cooldown > 0:
                blink_cooldown -= 1

            if face_results and face_results.face_landmarks:
                face_landmarks = DummyLandmarkList(face_results.face_landmarks[0])
                ear_value = compute_ear(face_landmarks)
                ear_filtered = exp_smooth_scalar(ear_filtered, ear_value, EAR_SMOOTH_FACTOR)
                if ear_baseline is None:
                    ear_baseline = ear_filtered
                if not eyes_closed and ear_filtered > ear_baseline * 0.92:
                    ear_baseline = exp_smooth_scalar(
                        ear_baseline, ear_filtered, EAR_BASELINE_SMOOTH
                    )

                close_threshold = max(BLINK_MIN_THRESHOLD, ear_baseline * BLINK_CLOSE_RATIO)
                open_threshold = max(close_threshold + 0.01, ear_baseline * BLINK_OPEN_RATIO)

                if eyes_closed:
                    if ear_filtered < open_threshold:
                        closed_frames += 1
                    else:
                        if closed_frames >= BLINK_CONSEC_FRAMES and blink_cooldown == 0:
                            blink_count += 1
                            blink_increment = 1
                            blink_cooldown = BLINK_REFRACTORY_FRAMES
                        eyes_closed = False
                        closed_frames = 0
                else:
                    if ear_filtered < close_threshold:
                        eyes_closed = True
                        closed_frames = 1

                drowsy = eyes_closed and closed_frames >= DROWSY_CLOSED_FRAMES
                looking_at_screen, _ = detect_looking_at_screen(face_landmarks, frame.shape)
                gaze_off_window.append(0 if looking_at_screen else 1)
                (
                    lock_start_time,
                    lock_start_points,
                    smoothed_bbox,
                ) = draw_face_lock_visual(
                    frame,
                    face_landmarks,
                    lock_start_time,
                    lock_start_points,
                    smoothed_bbox,
                )

            else:
                gaze_off_window.append(1)
                closed_frames = 0
                eyes_closed = False
                lock_start_time = None
                lock_start_points = None
                smoothed_bbox = None

            gaze_off_percent = (sum(gaze_off_window) * 100.0) / max(1, len(gaze_off_window))
            attention_raw = 100.0 - gaze_off_percent
            attention_percent = exp_smooth_scalar(
                attention_percent, attention_raw, ATTENTION_SMOOTH_FACTOR
            )
            now_ts = time.monotonic()
            signal_aggregator.update(
                now_ts=now_ts,
                ear=ear_value,
                posture=posture_score,
                looking_at_screen=looking_at_screen,
                drowsy=drowsy,
                blink_increment=blink_increment,
            )
            rolling_snapshot = signal_aggregator.snapshot(now_ts)
            ai_output_view = get_vision_agent_state()

            state_label = classify_cognitive_state(attention_percent, drowsy)
            confidence = estimate_state_confidence(
                attention_percent=attention_percent,
                posture_score=posture_score,
                drowsy=drowsy,
            )
            ai_state_text = str(ai_output_view.get("state", "")).strip()
            if not ai_state_text or ai_state_text in {"Analyzing", "Unavailable"}:
                ai_state_text = state_label
            ai_reason_text = str(ai_output_view.get("reason", "")).strip()
            if not ai_reason_text:
                ai_reason_text = "Live cognitive metrics active."
            risk_label = compute_risk_label(attention_percent, posture_score, drowsy)
            snapshot_text = format_cognitive_snapshot(rolling_snapshot)

            publish_tools_metrics(
                {
                    "attention": float(np.clip(attention_percent, 0.0, 100.0)),
                    "gaze_off": float(np.clip(gaze_off_percent, 0.0, 100.0)),
                    "blink_rate": float(rolling_snapshot.get("blink_rate", 0.0)),
                    "posture_score": float(np.clip(posture_score, 0.0, 1.0)),
                    "drowsy_events": int(rolling_snapshot.get("drowsy_events", 0)),
                    "fatigue_detected": bool(drowsy),
                    "state": state_label,
                    "confidence": confidence,
                    "ai_state": ai_state_text,
                    "reason": ai_reason_text,
                    "secondary_reason": f"Blink {float(rolling_snapshot.get('blink_rate', 0.0)):.1f}/min | GazeOff {float(np.clip(gaze_off_percent, 0.0, 100.0)):.1f}%",
                    "risk": risk_label,
                    "snapshot": snapshot_text,
                    "latency_ms": float(ai_output_view.get("latency_ms", 0.0)),
                    "latency_avg_ms": float(ai_output_view.get("latency_avg_ms", ai_output_view.get("latency_ms", 0.0))),
                    "fps": float(ai_output_view.get("fps", 0.0)),
                    "model": str(ai_output_view.get("model", "Unknown")),
                    "timestamp": str(ai_output_view.get("timestamp", "")),
                }
            )
            if (now_ts - last_bridge_write_ts) >= METRICS_BRIDGE_WRITE_INTERVAL_SECONDS:
                try:
                    write_metrics_bridge(get_tools_metrics())
                    last_bridge_write_ts = now_ts
                except Exception:
                    # Keep vision loop stable even if filesystem writes fail.
                    pass

            hud_bottom = draw_hud(
                frame,
                ai_output_view,
                posture_score,
                drowsy,
                attention_percent,
                face_present=face_detected,
            )
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
