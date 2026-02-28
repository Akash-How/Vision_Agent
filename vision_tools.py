"""Vision Agent tool wrappers for CogniGuard runtime metrics.

These functions are intentionally lightweight and only read already-computed
metrics exported by main.py (no camera or processing logic).
"""

from typing import Any, Dict


_metrics_reader = None


def _read_metrics() -> Dict[str, Any]:
    global _metrics_reader
    try:
        if _metrics_reader is None:
            from main import get_tools_metrics

            _metrics_reader = get_tools_metrics
        return dict(_metrics_reader())
    except Exception:
        return {}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def get_attention_metrics() -> Dict[str, float]:
    metrics = _read_metrics()
    return {
        "attention": _to_float(metrics.get("attention", 0.0)),
        "gaze_off": _to_float(metrics.get("gaze_off", 0.0)),
        "blink_rate": _to_float(metrics.get("blink_rate", 0.0)),
    }


def get_posture_score() -> Dict[str, float]:
    metrics = _read_metrics()
    return {"posture_score": _to_float(metrics.get("posture_score", 0.0))}


def get_fatigue_metrics() -> Dict[str, Any]:
    metrics = _read_metrics()
    return {
        "drowsy_events": _to_int(metrics.get("drowsy_events", 0)),
        "fatigue_detected": bool(metrics.get("fatigue_detected", False)),
    }


def get_current_state() -> Dict[str, Any]:
    metrics = _read_metrics()
    state = str(metrics.get("state", "Distracted"))
    if state not in {"Focused", "Fatigued", "Distracted"}:
        state = "Distracted"
    confidence = _to_float(metrics.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    return {"state": state, "confidence": confidence}
