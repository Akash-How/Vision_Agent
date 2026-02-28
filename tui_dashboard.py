import time
import json
import os
from datetime import datetime
from collections import deque

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
import rich.box as box


TREND_POINTS = 50
TREND_HEIGHT = 10
DATA_REFRESH_SECONDS = 1.0
UI_FRAME_SECONDS = 0.10
METRICS_BRIDGE_PATH = os.environ.get(
    "COGNIGUARD_METRICS_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cogniguard_metrics.json"),
)
VISION_STATE_PATH = os.environ.get(
    "COGNIGUARD_AGENT_STATE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_state.json"),
)
VISION_STATE_STALE_SECONDS = 10.0
METRICS_STALE_SECONDS = float(
    os.environ.get("COGNIGUARD_METRICS_STALE_SECONDS", "5.0")
)


def _as_float(data, key, default=0.0):
    try:
        return float(data.get(key, default))
    except Exception:
        return float(default)


def _as_int(data, key, default=0):
    try:
        return int(data.get(key, default))
    except Exception:
        return int(default)


def _as_str(data, key, default=""):
    value = data.get(key, default)
    if value is None:
        return default
    return str(value)


def _spread_line(parts, width):
    if not parts:
        return " " * max(1, width)
    columns = len(parts)
    col_w = max(1, width // columns)
    line = "".join(str(part)[:col_w].center(col_w) for part in parts)
    return line[:width].ljust(width)


def _state_to_attention(state, confidence):
    s = str(state).strip().title()
    c = max(0.0, min(1.0, float(confidence)))
    if s == "Focused":
        return 78.0 + (22.0 * c)
    if s == "Fatigued":
        return 40.0 + (15.0 * c)
    if s == "Distracted":
        return 28.0 + (22.0 * (1.0 - c))
    return 50.0


def _state_to_risk(state):
    s = str(state).strip().title()
    if s == "Fatigued":
        return "High"
    if s == "Distracted":
        return "Medium"
    return "Low"


def _parse_timestamp_to_epoch(value, fallback=0.0):
    try:
        if value is None:
            return float(fallback)
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return float(fallback)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return float(datetime.fromisoformat(text).timestamp())
    except Exception:
        return float(fallback)


def _scroll_text(message, offset, width):
    if width <= 0:
        return "", offset
    base = f"{message}   "
    repeats = (width // max(1, len(base))) + 3
    tape = base * repeats
    start = offset % len(base)
    window = tape[start : start + width]
    return window.ljust(width), (offset + 1) % len(base)


def _build_multi_trend(series_dict, width=100, height=15):
    if width <= 0 or height <= 0:
        return Group()

    chart = [[(" ", "black") for _ in range(width)] for _ in range(height)]

    # Draw background grid
    for r in [0, height // 4, height // 2, (3 * height) // 4, height - 1]:
        for c in range(width):
            chart[r][c] = ("-", "grey15")

    labels = []

    # Plot each series
    for name, (values, color) in series_dict.items():
        if not values:
            values = [0.0]
        series = list(values)[-width:]
        if len(series) < width:
            series = [series[0]] * (width - len(series)) + series

        vmin, vmax = min(series), max(series)
        span = vmax - vmin
        if span <= 1e-9:
            span = 1.0

        # normalize to 0..height-1
        levels = [int(round((v - vmin) / span * (height - 1))) for v in series]
        rows = [height - 1 - lvl for lvl in levels]

        # Draw raw points
        for x in range(width):
            if x < len(rows):
                y = rows[x]
                if 0 <= y < height:
                    chart[y][x] = ("•", color)

        # Interpolate between points
        for x in range(1, len(rows)):
            x0, y0 = x - 1, rows[x - 1]
            x1, y1 = x, rows[x]
            dx = x1 - x0
            dy = y1 - y0
            steps = max(abs(dx), abs(dy), 1)
            for s in range(1, steps):
                cx = x0 + int(round(dx * s / steps))
                cy = y0 + int(round(dy * s / steps))
                if 0 <= cy < height and 0 <= cx < width:
                    chart[cy][cx] = ("•", color)

        labels.append((name, color, series[-1]))

    lines = []
    
    # Legend
    legend_text = Text()
    legend_text.append("  ")
    for name, color, last_val in labels:
        legend_text.append(f"■ {name} ", style=color)
        legend_text.append(f"{last_val:5.1f}   ", style="white")
    lines.append(legend_text)
    lines.append(Text(""))

    # Render chart rows
    for r in range(height):
        row_text = Text()
        pct = 100 - int(r / max(1, height - 1) * 100)
        row_text.append(f"{pct:3d} ", style="grey50")
        row_text.append("│ ", style="grey15")

        for ch, color in chart[r]:
            row_text.append(ch, style=color)
        lines.append(row_text)

    # X-axis
    x_axis = Text("    └", style="grey15")
    x_axis.append("─" * width, style="grey15")
    lines.append(x_axis)

    return Group(*lines)


def _build_comparison_table(metrics, prev_metrics=None):
    if prev_metrics is None:
        prev_metrics = metrics
        
    table = Table(show_edge=False, box=box.SIMPLE_HEAD, expand=True, pad_edge=False, row_styles=["none", "grey15"])
    table.add_column("Name", style="bold white", ratio=2)
    table.add_column("Last Value", justify="right", style="cyan")
    table.add_column("Chg Pct (Avg)", justify="right", style="white")
    table.add_column("Tolerance", justify="right", style="white")
    table.add_column("Status", justify="right", style="bold")
    table.add_column("AI Signal", justify="right", style="white")

    # Attention Row
    attn = _as_float(metrics, 'attention', 100.0)
    p_attn = _as_float(prev_metrics, 'attention', 100.0)
    chg = attn - p_attn
    chg_str = f"[green]+{chg:.1f}%[/]" if chg >= 0 else f"[red]{chg:.1f}%[/]"
    status_fmt = "[green]OK[/]" if attn > 60 else "[red]WARN[/]"
    table.add_row(
        " Attention Score", 
        f"{attn:6.2f}", 
        chg_str, 
        "> 60% Valid", 
        status_fmt, 
        _as_str(metrics, "ai_state", "N/A")
    )

    # Blink Rate Row
    br = _as_float(metrics, 'blink_rate', 15.0)
    p_br = _as_float(prev_metrics, 'blink_rate', 15.0)
    b_chg = br - p_br
    b_chg_str = f"[red]+{b_chg:.1f}[/]" if b_chg > 0 else f"[green]{b_chg:.1f}[/]"
    b_status = "[red]HIGH[/]" if br > 25 else ("[blue]LOW[/]" if br < 8 else "[green]OK[/]")
    table.add_row(
        " Blink Rate", 
        f"{br:6.2f}", 
        b_chg_str, 
        "8.0 - 25.0", 
        b_status, 
        _as_str(metrics, "reason", "N/A")[:20]
    )

    # Posture Row
    posture = _as_float(metrics, "posture_score", _as_float(metrics, "posture", 0.0))
    p_posture = _as_float(prev_metrics, "posture_score", _as_float(prev_metrics, "posture", 0.0))
    p_chg = posture - p_posture
    p_chg_str = f"[green]+{p_chg:.2f}[/]" if p_chg > 0 else f"[red]{p_chg:.2f}[/]"
    p_status = "[green]OK[/]" if posture > -0.5 else "[red]WARN[/]"
    table.add_row(
        " Posture Index", 
        f"{posture:6.2f}", 
        p_chg_str, 
        "> -0.50", 
        p_status, 
        "VisionAgent"
    )
    
    # Gaze Row
    gaze = _as_float(metrics, 'gaze_off', 0.0)
    p_gaze = _as_float(prev_metrics, 'gaze_off', 0.0)
    g_chg = gaze - p_gaze
    g_chg_str = f"[red]+{g_chg:.1f}%[/]" if g_chg > 0 else f"[green]{g_chg:.1f}%[/]"
    g_status = "[red]WARN[/]" if gaze > 30 else "[green]OK[/]"
    table.add_row(
        " Gaze Off Ratio", 
        f"{gaze:6.2f}", 
        g_chg_str, 
        "< 30.0%", 
        g_status, 
        f"Conf: {_as_float(metrics, 'confidence', 0.0)*100:.0f}%"
    )

    return Panel(
        table,
        title=" PEER COMPARISON / REALTIME SIGNAL ",
        title_align="left",
        border_style="orange1",
        box=box.SQUARE,
        padding=(0, 1),
    )


def _build_layout(width):
    layout = Layout(name="root")
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="ticker", size=1),
    )
    layout["body"].split_column(
        Layout(name="main_chart", ratio=1),
        Layout(name="comparison_table", size=10),
    )
    return layout


def run_dashboard(metrics):
    """
    Run realtime professional TUI dashboard.

    `metrics` can be:
    - dict (mutable values can be updated externally)
    - callable returning dict
    """
    console = Console()
    layout = _build_layout(console.size.width)

    attention_history = deque(maxlen=TREND_POINTS)
    blink_history = deque(maxlen=300)
    posture_history = deque(maxlen=300)
    ticker_offset = 0

    last_frame = time.time()
    last_data_refresh = 0.0
    ui_fps = 0.0
    ui_latency_ms = 0.0
    current = {}
    prev_metrics = None

    with Live(layout, console=console, screen=True, refresh_per_second=10):
        try:
            while True:
                now = time.time()
                dt = max(1e-6, now - last_frame)
                last_frame = now
                ui_fps = 1.0 / dt
                ui_latency_ms = dt * 1000.0

                if now - last_data_refresh >= DATA_REFRESH_SECONDS:
                    new_curr = metrics() if callable(metrics) else dict(metrics)
                    if current:
                        prev_metrics = dict(current)
                    current = new_curr
                    
                    attention = _as_float(current, "attention", 0.0)
                    blink_rate = _as_float(current, "blink_rate", 0.0)
                    posture = _as_float(current, "posture_score", _as_float(current, "posture", 0.0))
                    attention_history.append(attention)
                    blink_history.append(blink_rate)
                    posture_history.append(posture)
                    last_data_refresh = now
                else:
                    attention = _as_float(current, "attention", 0.0)
                    blink_rate = _as_float(current, "blink_rate", 0.0)
                    posture = _as_float(current, "posture_score", _as_float(current, "posture", 0.0))

                clock = time.strftime("%H:%M:%S", time.localtime(now))
                ai_state_raw = _as_str(current, 'ai_state', 'Unknown')
                inf_fps = _as_float(current, "fps", 0.0)
                inf_latency_ms = _as_float(
                    current,
                    "latency_avg_ms",
                    _as_float(current, "latency_ms", 0.0),
                )
                model_name = _as_str(current, "model", "Unknown")
                
                # Build Row 1: Top Red Bar
                r1_left = " COGNIGUARD TERMINAL"
                r1_right = "Live Analytics Platform  * LIVE "
                spaces_r1 = max(1, console.size.width - len(r1_left) - len(r1_right))
                row1 = Text(r1_left + (" " * spaces_r1) + r1_right, style="bold white on dark_red")

                # Build Row 2: Sub Data Bar
                r2_txt = (
                    f" VISN US  $ {ai_state_raw.upper()}  {attention:5.1f}%"
                    f"  |  At {clock}"
                    f"  |  INF_FPS {inf_fps:4.1f}"
                    f"  LAT {inf_latency_ms/1000.0:4.1f}s"
                    f"  |  Model {model_name}"
                    f"  |  Risk: {_as_str(current, 'risk', 'Low')} "
                )
                row2 = Text(r2_txt.ljust(console.size.width), style="bold bright_green on black")

                # Build Row 3: Tabs
                tabs = [
                    ("[0] HOME", "white on dark_blue"),
                    ("[1] OVERVIEW", "white on dark_blue"),
                    ("[2] AI ANALYSIS", "white on dark_blue"),
                    ("[3] SIGNALS", "white on dark_blue"),
                    ("[4] LIVE TRENDS", "bold black on bright_yellow"),
                    ("[5] OWNERSHIP", "white on dark_blue"),
                ]
                row3 = Text()
                for i, (text, style) in enumerate(tabs):
                    row3.append(f" {text} ", style=style)
                    if i < len(tabs) - 1:
                        row3.append(" | ", style="bright_black on dark_blue")
                
                padding_r3 = max(0, console.size.width - len(row3.plain))
                row3.append(" " * padding_r3, style="white on dark_blue")

                header_group = Group(row1, row2, row3)
                layout["header"].update(header_group)

                ticker_items = [
                    f"VISN {attention:5.1f}  {_as_str(current, 'ai_state', 'Unknown')}",
                    f"BLK {blink_rate:5.1f} {'+0.1' if blink_rate > 15 else '-0.1'} (+0.25%)",
                    f"POS {posture:5.2f} {'+0.05' if posture > 0 else '-0.02'} (-0.11%)",
                    f"ATTN {attention:5.1f} {'+1.2' if attention > 80 else '-0.5'} (+1.52%)",
                    f"LNCY {inf_latency_ms/1000.0:4.1f}s (avg)",
                    f"INFPS {inf_fps:4.1f}  RENDER {ui_fps:4.1f}",
                    f"DRWSY {_as_int(current, 'drowsy_events', 0)} +0 (+0.0%)",
                ]
                # Alternate green and red styles to look like moving ticker numbers
                t_parts = []
                for i, tm in enumerate(ticker_items):
                    color = "bright_green" if i % 2 == 0 else "red"
                    t_parts.append(f"[orange1]O[/] {tm.split(' ')[0]} {tm.split(' ')[1]} [{color}]{' '.join(tm.split(' ')[2:])}[/]")
                
                ticker_msg = "   ".join(t_parts)
                ticker_line, ticker_offset = _scroll_text(
                    ticker_msg, ticker_offset, console.size.width
                )
                
                # We can't use style in Text if we embedded markup, so we render via rich core formatting
                layout["ticker"].update(Panel(ticker_line, box=box.SIMPLE, padding=(0,0)))

                # Update main_chart with multi-metric trend
                chart_w_pts = max(10, console.size.width - 15)
                chart_h_pts = max(5, console.size.height - 24)
                
                series_data = {
                    "ATTN (%)": (attention_history, "bright_cyan"),
                    "BLNK (/m)": (blink_history, "bright_red"),
                    "POST (idx)": (posture_history, "bright_green"),
                }
                
                trend_group = _build_multi_trend(
                    series_data, 
                    width=chart_w_pts, 
                    height=chart_h_pts
                )
                
                layout["main_chart"].update(
                    Panel(
                        trend_group, 
                        title=" [4] REL VALUE ", 
                        title_align="center",
                        border_style="orange1", 
                        box=box.SQUARE
                    )
                )

                # Update comparison_table
                layout["comparison_table"].update(
                    _build_comparison_table(current, prev_metrics)
                )

                time.sleep(UI_FRAME_SECONDS)
        except KeyboardInterrupt:
            return


def _build_live_metrics_provider(path=METRICS_BRIDGE_PATH, vision_path=VISION_STATE_PATH):
    defaults = {
        "attention": 0.0,
        "blink_rate": 0.0,
        "posture_score": 0.0,
        "gaze_off": 0.0,
        "drowsy_events": 0,
        "ai_state": "Waiting",
        "confidence": 0.0,
        "reason": f"Waiting for {path}",
        "secondary_reason": "Start main.py to stream metrics.",
        "risk": "Unknown",
        "snapshot": "No snapshot",
        "vision_agent": False,
        "model": "Unknown",
        "latency_ms": 0.0,
        "latency_avg_ms": 0.0,
        "fps": 0.0,
        "timestamp": "",
    }

    def provider():
        out = dict(defaults)
        metrics_fresh = False
        vision_fresh = False

        try:
            if os.path.exists(path):
                metrics_mtime = os.path.getmtime(path)
                metrics_age = max(0.0, time.time() - metrics_mtime)
                metrics_fresh = metrics_age <= METRICS_STALE_SECONDS
                if metrics_fresh:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        out.update(data)
                        # Normalization for dashboard keys.
                        if "posture_score" not in out and "posture" in out:
                            out["posture_score"] = _as_float(out, "posture", 0.0)
                        if "ai_state" not in out and "state" in out:
                            out["ai_state"] = _as_str(out, "state", "Unknown")
        except Exception:
            metrics_fresh = False

        try:
            if os.path.exists(vision_path):
                # Always read latest vision_state.json every refresh cycle.
                with open(vision_path, "r", encoding="utf-8") as f:
                    v = json.load(f)
                if isinstance(v, dict):
                    ts_epoch = _parse_timestamp_to_epoch(
                        v.get("timestamp", None),
                        fallback=float(v.get("timestamp_epoch", os.path.getmtime(vision_path))),
                    )
                    vision_fresh = (time.time() - ts_epoch) <= VISION_STATE_STALE_SECONDS
                    state = _as_str(v, "state", _as_str(out, "ai_state", "Unknown"))
                    conf = _as_float(v, "confidence", _as_float(out, "confidence", 0.0))
                    reason = _as_str(v, "reason", _as_str(out, "reason", "N/A"))
                    out["ai_state"] = state
                    out["state"] = state
                    out["confidence"] = conf
                    out["reason"] = reason
                    out["model"] = _as_str(v, "model", _as_str(out, "model", "Unknown"))
                    out["latency_ms"] = _as_float(v, "latency_ms", _as_float(out, "latency_ms", 0.0))
                    out["latency_avg_ms"] = _as_float(
                        v, "latency_avg_ms", _as_float(v, "latency_ms", _as_float(out, "latency_avg_ms", 0.0))
                    )
                    out["fps"] = _as_float(v, "fps", _as_float(out, "fps", 0.0))
                    out["timestamp"] = _as_str(v, "timestamp", "")
                    out["vision_agent"] = bool(v.get("vision_agent", True)) and vision_fresh
                    out["risk"] = _state_to_risk(state)

                    if not vision_fresh:
                        out["vision_agent"] = False
                        out["reason"] = "Vision agent state is stale."
        except Exception:
            vision_fresh = False

        # If metrics are stale but vision is fresh, prefer vision-derived values.
        if vision_fresh and not metrics_fresh:
            state = _as_str(out, "ai_state", _as_str(out, "state", "Unknown"))
            conf = _as_float(out, "confidence", 0.0)
            attention = float(_state_to_attention(state, conf))
            out["attention"] = attention
            out["gaze_off"] = max(0.0, 100.0 - attention)
            # Keep these stable and explicit in vision-only mode.
            out["blink_rate"] = _as_float(out, "blink_rate", 0.0)
            out["posture_score"] = _as_float(out, "posture_score", 0.0)
            out["drowsy_events"] = _as_int(out, "drowsy_events", 0)
            out["risk"] = _state_to_risk(state)
            if not _as_str(out, "snapshot", "").strip():
                out["snapshot"] = "Vision agent only mode"

        if (not vision_fresh) and (not metrics_fresh):
            if os.path.exists(vision_path):
                out["vision_agent"] = False
                out["reason"] = "Vision agent state is stale."
            else:
                out["vision_agent"] = False
                out["reason"] = "Waiting for live data."
                out["ai_state"] = "Waiting"
        return out

    return provider


if __name__ == "__main__":
    run_dashboard(_build_live_metrics_provider())
