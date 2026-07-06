"""
dashboard.py
============

All UI logic for the Vehicle Authentication Command Center.

This module renders a dark, futuristic, Tesla / NVIDIA-GTC inspired
operations dashboard for an existing autonomous-vehicle authentication
backend. It communicates with that backend EXCLUSIVELY over HTTP
(`POST /authenticate`) and never touches backend code, models, or engines.

Architecture
------------
The dashboard is built from small, single-responsibility, reusable
components (`Theme`, `SystemMonitor`, `APIClient`, `PipelineVisualizer`,
`ScoreGauge`, `DecisionBadge`, individual result cards, `TopBar`,
`InputPanel`) that are orchestrated by the top-level `AuthDashboard`
class. Every component that renders *optional* backend data exposes a
`.hide()` / `.show()` pair so the layout degrades gracefully when a
field is absent from the API response, rather than raising an error.

This file is immediately runnable against the described backend:
    POST http://localhost:8000/authenticate
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import httpx
import plotly.graph_objects as go
from nicegui import ui

try:  # pragma: no cover - optional dependency, degrades gracefully
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


# =============================================================================
# Theme
# =============================================================================
class Theme:
    """Centralized design tokens for the dark, futuristic dashboard.

    Kept as a single source of truth so every component references the same
    palette, spacing, and glow effects instead of hard-coding colors inline.
    """

    BG_VOID: str = "#05070A"
    BG_BASE: str = "#0A0E14"
    BG_PANEL: str = "#0F151D"
    BG_CARD: str = "#131A24"
    BORDER: str = "#1E2733"

    TEXT_PRIMARY: str = "#E8EEF5"
    TEXT_SECONDARY: str = "#8A97A8"
    TEXT_MUTED: str = "#5A6472"

    ACCENT_CYAN: str = "#00E5FF"
    ACCENT_GREEN: str = "#76B900"  # NVIDIA green
    ACCENT_AMBER: str = "#FFB020"
    ACCENT_RED: str = "#FF3B5C"
    ACCENT_VIOLET: str = "#8B5CF6"

    STATUS_COLORS: dict[str, str] = {
        "ALLOW": "#76B900",
        "OTP_REQUIRED": "#FFB020",
        "VOICE_REAUTH": "#00E5FF",
        "MANUAL_REVIEW": "#8B5CF6",
        "BLOCK": "#FF3B5C",
        "UNKNOWN": "#5A6472",
    }

    FONT_DISPLAY: str = "'Orbitron', 'Rajdhani', sans-serif"
    FONT_BODY: str = "'Rajdhani', 'Inter', sans-serif"
    FONT_MONO: str = "'JetBrains Mono', 'Fira Code', monospace"

    @classmethod
    def status_color(cls, status: Optional[str]) -> str:
        """Return the accent color associated with a decision status."""
        if not status:
            return cls.STATUS_COLORS["UNKNOWN"]
        return cls.STATUS_COLORS.get(status.upper(), cls.STATUS_COLORS["UNKNOWN"])

    @classmethod
    def inject_global_styles(cls) -> None:
        """Inject fonts, CSS variables, glow effects, and keyframes once."""
        ui.add_head_html(
            '<link rel="preconnect" href="https://fonts.googleapis.com">'
            '<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900'
            '&family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" '
            'rel="stylesheet">'
        )
        ui.add_css(
            f"""
            body {{
                background: radial-gradient(ellipse at top, {cls.BG_BASE} 0%, {cls.BG_VOID} 70%);
                font-family: {cls.FONT_BODY};
                color: {cls.TEXT_PRIMARY};
            }}
            .avc-card {{
                background: linear-gradient(180deg, {cls.BG_CARD} 0%, {cls.BG_PANEL} 100%);
                border: 1px solid {cls.BORDER};
                border-radius: 16px;
                padding: 14px 16px;
                transition: box-shadow 0.4s ease, border-color 0.4s ease, transform 0.25s ease;
            }}
            .avc-card:hover {{
                border-color: {cls.ACCENT_CYAN}55;
            }}
            .avc-glow-cyan {{ box-shadow: 0 0 22px {cls.ACCENT_CYAN}33, inset 0 0 30px {cls.ACCENT_CYAN}0d; }}
            .avc-glow-green {{ box-shadow: 0 0 22px {cls.ACCENT_GREEN}44, inset 0 0 30px {cls.ACCENT_GREEN}0d; }}
            .avc-glow-red {{ box-shadow: 0 0 26px {cls.ACCENT_RED}55, inset 0 0 30px {cls.ACCENT_RED}0d; }}
            .avc-glow-amber {{ box-shadow: 0 0 22px {cls.ACCENT_AMBER}44, inset 0 0 30px {cls.ACCENT_AMBER}0d; }}
            .avc-glow-violet {{ box-shadow: 0 0 22px {cls.ACCENT_VIOLET}44, inset 0 0 30px {cls.ACCENT_VIOLET}0d; }}

            .avc-title {{
                font-family: {cls.FONT_DISPLAY};
                letter-spacing: 0.08em;
            }}
            .avc-mono {{ font-family: {cls.FONT_MONO}; }}

            @keyframes avc-pulse {{
                0%   {{ box-shadow: 0 0 6px currentColor; opacity: 0.75; }}
                50%  {{ box-shadow: 0 0 26px currentColor; opacity: 1; }}
                100% {{ box-shadow: 0 0 6px currentColor; opacity: 0.75; }}
            }}
            .avc-pulse {{ animation: avc-pulse 1.4s ease-in-out infinite; }}

            @keyframes avc-flow {{
                0%   {{ stroke-dashoffset: 24; }}
                100% {{ stroke-dashoffset: 0; }}
            }}
            .avc-flow-line {{ animation: avc-flow 0.6s linear infinite; }}

            @keyframes avc-fade-in {{
                from {{ opacity: 0; transform: translateY(6px); }}
                to   {{ opacity: 1; transform: translateY(0); }}
            }}
            .avc-fade-in {{ animation: avc-fade-in 0.35s ease-out; }}

            .avc-node {{
                border-radius: 14px;
                border: 1.5px solid {cls.BORDER};
                background: {cls.BG_PANEL};
                transition: all 0.35s ease;
            }}
            .avc-node-idle {{ opacity: 0.55; }}
            .avc-node-active {{
                border-color: {cls.ACCENT_CYAN};
                box-shadow: 0 0 24px {cls.ACCENT_CYAN}66;
                opacity: 1;
            }}
            .avc-node-done {{
                border-color: {cls.ACCENT_GREEN};
                box-shadow: 0 0 18px {cls.ACCENT_GREEN}55;
                opacity: 1;
            }}
            .avc-node-error {{
                border-color: {cls.ACCENT_RED};
                box-shadow: 0 0 18px {cls.ACCENT_RED}55;
                opacity: 1;
            }}
            .avc-scrollbar::-webkit-scrollbar {{ width: 8px; height: 8px; }}
            .avc-scrollbar::-webkit-scrollbar-track {{ background: transparent; }}
            .avc-scrollbar::-webkit-scrollbar-thumb {{ background: {cls.BORDER}; border-radius: 8px; }}
            """
        )


# =============================================================================
# Data helpers
# =============================================================================
def safe_get(source: Any, *keys: str, default: Any = None) -> Any:
    """Safely walk a chain of dictionary keys, returning `default` if missing.

    Args:
        source: The (possibly nested) dictionary to read from.
        *keys: Sequence of keys to traverse in order.
        default: Value returned if any key is missing or `source` isn't a dict.
    """
    current = source
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current if current is not None else default


def find_numeric_by_keywords(source: Any, keywords: tuple[str, ...]) -> Optional[float]:
    """Recursively search a nested dict for a numeric value whose key matches
    any of the given keywords (case-insensitive substring match).

    Used to *best-effort* surface optional per-stage timing information that
    a backend may embed inside `audit_log` under an unspecified key name,
    without assuming a rigid schema. Returns `None` if nothing matches, so
    callers can hide the corresponding widget gracefully.
    """
    if isinstance(source, dict):
        for key, value in source.items():
            key_lower = str(key).lower()
            if isinstance(value, (int, float)) and any(kw in key_lower for kw in keywords):
                return float(value)
        for value in source.values():
            found = find_numeric_by_keywords(value, keywords)
            if found is not None:
                return found
    return None


def find_list_by_keywords(source: Any, keywords: tuple[str, ...]) -> Optional[list]:
    """Recursively search a nested dict for a list value whose key matches
    any of the given keywords. Returns `None` when nothing matches."""
    if isinstance(source, dict):
        for key, value in source.items():
            key_lower = str(key).lower()
            if isinstance(value, list) and any(kw in key_lower for kw in keywords):
                return value
        for value in source.values():
            found = find_list_by_keywords(value, keywords)
            if found is not None:
                return found
    return None


def find_dict_by_keywords(source: Any, keywords: tuple[str, ...]) -> Optional[dict]:
    """Recursively search a nested dict for a dict value whose key matches
    any of the given keywords. Returns `None` when nothing matches."""
    if isinstance(source, dict):
        for key, value in source.items():
            key_lower = str(key).lower()
            if isinstance(value, dict) and any(kw in key_lower for kw in keywords):
                return value
        for value in source.values():
            found = find_dict_by_keywords(value, keywords)
            if found is not None:
                return found
    return None


# =============================================================================
# System monitor (Top Bar / Performance Panel telemetry)
# =============================================================================
class SystemMonitor:
    """Best-effort local host telemetry: CPU, memory, and GPU utilization.

    All values degrade gracefully to `None` when the relevant library or
    hardware isn't available, so callers can render "N/A" instead of failing.
    """

    def __init__(self) -> None:
        self._gpu_handle = None
        self._pynvml = None
        try:  # pragma: no cover - optional dependency, hardware-dependent
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
        except Exception:
            self._gpu_handle = None
            self._pynvml = None

    def cpu_percent(self) -> Optional[float]:
        """Return current CPU utilization percentage, or `None` if unavailable."""
        if psutil is None:
            return None
        return psutil.cpu_percent(interval=None)

    def memory_percent(self) -> Optional[float]:
        """Return current memory utilization percentage, or `None` if unavailable."""
        if psutil is None:
            return None
        return psutil.virtual_memory().percent

    def gpu_percent(self) -> Optional[float]:
        """Return current GPU utilization percentage, or `None` if no GPU/driver."""
        if self._gpu_handle is None or self._pynvml is None:
            return None
        try:  # pragma: no cover - hardware-dependent
            usage = self._pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            return float(usage.gpu)
        except Exception:
            return None

    def has_gpu(self) -> bool:
        """Whether a queryable GPU was detected on this host."""
        return self._gpu_handle is not None


# =============================================================================
# API client
# =============================================================================
@dataclass
class AuthResult:
    """Outcome of a single call to the authentication backend."""

    ok: bool
    status_code: Optional[int] = None
    response: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: float = 0.0


class APIClient:
    """Thin async HTTP client wrapping the backend's `/authenticate` endpoint.

    The dashboard NEVER talks to the Authentication Network, Intent Engine,
    Risk Engine, Policy Engine, or Decision Engine directly -- only to this
    single HTTP endpoint, per the integration contract.
    """

    def __init__(self, backend_url: str, timeout_seconds: float = 30.0) -> None:
        self._endpoint = backend_url.rstrip("/") + "/authenticate"
        self._timeout = timeout_seconds

    async def authenticate(self, payload: dict[str, Any]) -> AuthResult:
        """POST the authentication request payload and return a normalized result."""
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._endpoint, json=payload)
            latency_ms = (time.perf_counter() - start) * 1000.0
            if response.status_code >= 400:
                return AuthResult(
                    ok=False,
                    status_code=response.status_code,
                    error=f"Backend returned HTTP {response.status_code}: {response.text[:300]}",
                    latency_ms=latency_ms,
                )
            return AuthResult(
                ok=True,
                status_code=response.status_code,
                response=response.json(),
                latency_ms=latency_ms,
            )
        except httpx.ConnectError:
            return AuthResult(
                ok=False,
                error=f"Could not connect to backend at {self._endpoint}. "
                "Is the authentication service running on port 8000?",
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        except httpx.TimeoutException:
            return AuthResult(
                ok=False,
                error="Request to backend timed out.",
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001 - surface any unexpected client error
            return AuthResult(
                ok=False,
                error=f"Unexpected client error: {exc}",
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )


# =============================================================================
# Pipeline visualizer (Center Panel)
# =============================================================================
class PipelineStage(str, Enum):
    """The six sequential stages of the authentication backend pipeline."""

    FEATURE_EXTRACTION = "Feature Extraction"
    AUTH_NETWORK = "Authentication Network"
    INTENT_ENGINE = "Intent Engine"
    RISK_ENGINE = "Risk Engine"
    POLICY_ENGINE = "Policy Engine"
    DECISION_ENGINE = "Decision Engine"


STAGE_ICONS: dict[PipelineStage, str] = {
    PipelineStage.FEATURE_EXTRACTION: "graphic_eq",
    PipelineStage.AUTH_NETWORK: "psychology",
    PipelineStage.INTENT_ENGINE: "chat",
    PipelineStage.RISK_ENGINE: "warning",
    PipelineStage.POLICY_ENGINE: "gavel",
    PipelineStage.DECISION_ENGINE: "flag",
}


class PipelineNode:
    """A single animated node within the pipeline visualizer."""

    def __init__(self, stage: PipelineStage) -> None:
        self.stage = stage
        self._card: Optional[ui.element] = None
        self._icon: Optional[ui.icon] = None
        self._label: Optional[ui.label] = None
        self._time_label: Optional[ui.label] = None
        self._build()

    def _build(self) -> None:
        with ui.column().classes("items-center gap-1 w-full") as wrapper:
            self._card = (
                ui.column()
                .classes("avc-node avc-node-idle items-center justify-center w-full py-3 gap-1")
            )
            with self._card:
                self._icon = ui.icon(STAGE_ICONS[self.stage], size="1.6rem").classes(
                    "text-[color:var(--avc-node-color,#5A6472)]"
                )
                self._label = ui.label(self.stage.value).classes(
                    "text-xs text-center avc-title"
                ).style(f"color: {Theme.TEXT_SECONDARY}")
                self._time_label = ui.label("").classes("text-[10px] avc-mono").style(
                    f"color: {Theme.TEXT_MUTED}"
                )
        self.wrapper = wrapper

    def set_idle(self) -> None:
        """Reset the node to its dim, pre-execution state."""
        self._card.classes(remove="avc-node-active avc-node-done avc-node-error avc-pulse")
        self._card.classes(add="avc-node-idle")
        self._time_label.set_text("")

    def set_active(self) -> None:
        """Mark the node as currently executing (glowing, pulsing)."""
        self._card.classes(remove="avc-node-idle avc-node-done avc-node-error")
        self._card.classes(add="avc-node-active avc-pulse")
        self._time_label.set_text("running…")

    def set_done(self, duration_ms: Optional[float]) -> None:
        """Mark the node as complete, optionally displaying its execution time."""
        self._card.classes(remove="avc-node-idle avc-node-active avc-pulse avc-node-error")
        self._card.classes(add="avc-node-done")
        self._time_label.set_text(f"{duration_ms:.0f} ms" if duration_ms is not None else "✓")

    def set_error(self) -> None:
        """Mark the node as failed."""
        self._card.classes(remove="avc-node-idle avc-node-active avc-pulse avc-node-done")
        self._card.classes(add="avc-node-error")
        self._time_label.set_text("failed")


class PipelineVisualizer:
    """Renders the six-stage pipeline as a horizontal row of glowing nodes
    connected by animated flow lines, per the required animation flow:

        Feature Extraction -> Authentication Network -> Intent Engine ->
        Risk Engine -> Policy Engine -> Decision Engine
    """

    STAGES: tuple[PipelineStage, ...] = (
        PipelineStage.FEATURE_EXTRACTION,
        PipelineStage.AUTH_NETWORK,
        PipelineStage.INTENT_ENGINE,
        PipelineStage.RISK_ENGINE,
        PipelineStage.POLICY_ENGINE,
        PipelineStage.DECISION_ENGINE,
    )

    def __init__(self) -> None:
        self.nodes: dict[PipelineStage, PipelineNode] = {}
        self._total_label: Optional[ui.label] = None
        self._build()

    def _connector(self) -> None:
        ui.icon("chevron_right", size="1.4rem").classes("self-center").style(
            f"color: {Theme.ACCENT_CYAN}88"
        )

    def _build(self) -> None:
        with ui.column().classes("avc-card w-full gap-3"):
            with ui.row().classes("items-center justify-between w-full"):
                ui.label("EXECUTION PIPELINE").classes("avc-title text-sm").style(
                    f"color: {Theme.TEXT_SECONDARY}; letter-spacing: 0.15em;"
                )
                self._total_label = ui.label("Total: —").classes("avc-mono text-sm").style(
                    f"color: {Theme.ACCENT_CYAN}"
                )
            with ui.row().classes("items-stretch justify-between w-full flex-nowrap gap-1"):
                for i, stage in enumerate(self.STAGES):
                    with ui.column().classes("flex-1 min-w-0"):
                        self.nodes[stage] = PipelineNode(stage)
                    if i < len(self.STAGES) - 1:
                        self._connector()

    def reset(self) -> None:
        """Return every node to its idle state and clear the total latency."""
        for node in self.nodes.values():
            node.set_idle()
        self._total_label.set_text("Total: —")

    def set_total(self, total_ms: float) -> None:
        """Display the overall round-trip latency once the request completes."""
        self._total_label.set_text(f"Total: {total_ms:.0f} ms")

    async def animate_pass(
        self, stage_times: dict[PipelineStage, Optional[float]], step_delay: float = 0.16
    ) -> None:
        """Animate the pipeline lighting up one stage at a time.

        This provides visual pacing for the request lifecycle. Each node is
        marked active, held briefly, then marked done -- displaying the real
        per-stage execution time when the backend supplies one, or a simple
        checkmark when it does not.
        """
        for stage in self.STAGES:
            self.nodes[stage].set_active()
            await asyncio.sleep(step_delay)
            self.nodes[stage].set_done(stage_times.get(stage))

    def set_error_at(self, stage: PipelineStage) -> None:
        """Mark a specific stage (and freeze all subsequent stages) as failed."""
        found = False
        for s in self.STAGES:
            if s == stage:
                found = True
                self.nodes[s].set_error()
            elif found:
                self.nodes[s].set_idle()


# =============================================================================
# Score gauges (animated Plotly indicators)
# =============================================================================
class ScoreGauge:
    """An animated Plotly gauge indicator for a single 0-1 (or 0-100) score."""

    def __init__(
        self,
        title: str,
        color: str,
        value_format: str = "percent",
        height: int = 170,
    ) -> None:
        self._title = title
        self._color = color
        self._value_format = value_format
        self._height = height
        self._plot: Optional[ui.plotly] = None
        self._container: Optional[ui.element] = None
        self._build()

    @staticmethod
    def _hex_to_rgba(hex_color: str, alpha: float) -> str:
        """Convert a `#RRGGBB` hex color to an `rgba(...)` string, since Plotly
        does not accept 8-digit hex-with-alpha color strings."""
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha})"

    def _make_figure(self, value: float) -> go.Figure:
        display_value = value * 100 if self._value_format == "percent" else value
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=display_value,
                number={"suffix": "%" if self._value_format == "percent" else "", "font": {"size": 26}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": Theme.TEXT_MUTED, "tickfont": {"size": 9}},
                    "bar": {"color": self._color},
                    "bgcolor": Theme.BG_PANEL,
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": self._hex_to_rgba(Theme.ACCENT_RED, 0.13)},
                        {"range": [40, 70], "color": self._hex_to_rgba(Theme.ACCENT_AMBER, 0.13)},
                        {"range": [70, 100], "color": self._hex_to_rgba(self._color, 0.13)},
                    ],
                },
                domain={"x": [0, 1], "y": [0, 1]},
            )
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": Theme.TEXT_PRIMARY, "family": Theme.FONT_MONO},
            margin={"l": 20, "r": 20, "t": 10, "b": 0},
            height=self._height,
            transition={"duration": 700, "easing": "cubic-in-out"},
        )
        return fig

    def _build(self) -> None:
        with ui.column().classes("items-center gap-0 flex-1 min-w-0") as container:
            self._container = container
            ui.label(self._title).classes("text-xs avc-title").style(
                f"color: {Theme.TEXT_SECONDARY}; letter-spacing: 0.1em;"
            )
            self._plot = ui.plotly(self._make_figure(0.0)).classes("w-full")

    def update(self, value: Optional[float]) -> None:
        """Animate the gauge to a new value in the [0, 1] range (or hide if None)."""
        if value is None:
            self.hide()
            return
        self.show()
        self._plot.figure = self._make_figure(max(0.0, min(1.0, value)))
        self._plot.update()

    def hide(self) -> None:
        """Hide this gauge gracefully when its backing value isn't provided."""
        self._container.visible = False

    def show(self) -> None:
        """Reveal this gauge once its backing value becomes available."""
        self._container.visible = True


# =============================================================================
# Decision badge
# =============================================================================
class DecisionBadge:
    """The large, animated, color-coded final-decision badge."""

    STATUS_ICONS: dict[str, str] = {
        "ALLOW": "check_circle",
        "OTP_REQUIRED": "sms",
        "VOICE_REAUTH": "record_voice_over",
        "MANUAL_REVIEW": "support_agent",
        "BLOCK": "block",
        "UNKNOWN": "help",
    }

    def __init__(self) -> None:
        self._container: Optional[ui.element] = None
        self._icon: Optional[ui.icon] = None
        self._status_label: Optional[ui.label] = None
        self._message_label: Optional[ui.label] = None
        self._build()

    def _build(self) -> None:
        with ui.column().classes("avc-card w-full items-center gap-2") as container:
            self._container = container
            ui.label("DECISION").classes("avc-title text-xs self-start").style(
                f"color: {Theme.TEXT_SECONDARY}; letter-spacing: 0.15em;"
            )
            self._icon = ui.icon("help", size="2.6rem")
            self._status_label = ui.label("AWAITING REQUEST").classes(
                "avc-title text-lg text-center"
            )
            self._message_label = ui.label("Submit an authentication request to begin.").classes(
                "text-sm text-center"
            ).style(f"color: {Theme.TEXT_SECONDARY}")

    def set_pending(self) -> None:
        """Show a neutral, in-progress state while the pipeline runs."""
        color = Theme.TEXT_MUTED
        self._container.classes(
            remove=" ".join(f"avc-glow-{c}" for c in ("cyan", "green", "amber", "red", "violet"))
        )
        self._icon.props(f'name="hourglass_top" color="{color}"')
        self._status_label.set_text("PROCESSING…")
        self._status_label.style(f"color: {color}")
        self._message_label.set_text("Running the authentication pipeline.")

    def set_result(self, status: Optional[str], message: Optional[str]) -> None:
        """Render the final decision with status-appropriate color and icon."""
        key = (status or "UNKNOWN").upper()
        color = Theme.status_color(key)
        icon_name = self.STATUS_ICONS.get(key, self.STATUS_ICONS["UNKNOWN"])
        glow_map = {
            "ALLOW": "avc-glow-green",
            "OTP_REQUIRED": "avc-glow-amber",
            "VOICE_REAUTH": "avc-glow-cyan",
            "MANUAL_REVIEW": "avc-glow-violet",
            "BLOCK": "avc-glow-red",
        }
        self._container.classes(
            remove=" ".join(f"avc-glow-{c}" for c in ("cyan", "green", "amber", "red", "violet"))
        )
        self._container.classes(add=f"{glow_map.get(key, '')} avc-fade-in")
        self._icon.props(f'name="{icon_name}" color="{color}"')
        self._status_label.set_text(key.replace("_", " "))
        self._status_label.style(f"color: {color}")
        self._message_label.set_text(message or "No additional message provided.")

    def set_error(self, error_text: str) -> None:
        """Render a client/network error distinctly from a backend decision."""
        self._container.classes(
            remove=" ".join(f"avc-glow-{c}" for c in ("cyan", "green", "amber", "red", "violet"))
        )
        self._container.classes(add="avc-glow-red avc-fade-in")
        self._icon.props(f'name="error" color="{Theme.ACCENT_RED}"')
        self._status_label.set_text("REQUEST FAILED")
        self._status_label.style(f"color: {Theme.ACCENT_RED}")
        self._message_label.set_text(error_text)


# =============================================================================
# Result cards (Right Panel)
# =============================================================================
class CollapsibleCard:
    """Base helper for a titled, glowing card that can hide itself gracefully
    when the backend does not return the data it depends on."""

    def __init__(self, title: str, icon: str) -> None:
        self._title = title
        self._icon = icon
        self.container: Optional[ui.element] = None
        self.body: Optional[ui.element] = None
        self._build_shell()

    def _build_shell(self) -> None:
        with ui.column().classes("avc-card w-full gap-2 avc-fade-in") as container:
            self.container = container
            with ui.row().classes("items-center gap-2"):
                ui.icon(self._icon, size="1.1rem").style(f"color: {Theme.ACCENT_CYAN}")
                ui.label(self._title).classes("avc-title text-sm").style(
                    f"color: {Theme.TEXT_SECONDARY}; letter-spacing: 0.1em;"
                )
            self.body = ui.column().classes("w-full gap-1")

    def hide(self) -> None:
        """Hide the entire card when its backing data is absent."""
        self.container.visible = False

    def show(self) -> None:
        """Reveal the card once backing data becomes available."""
        self.container.visible = True

    def clear(self) -> None:
        self.body.clear()


class IntentCard(CollapsibleCard):
    """Displays Intent Engine output: intent, beneficiary, amount, etc."""

    def __init__(self) -> None:
        super().__init__("Intent", "chat_bubble")

    def update(self, intent_data: Optional[dict]) -> None:
        self.clear()
        if not intent_data:
            self.hide()
            return
        self.show()
        rows = [
            ("Intent", intent_data.get("intent")),
            ("Amount", self._format_amount(intent_data)),
            ("Beneficiary", intent_data.get("beneficiary")),
            ("Beneficiary Type", intent_data.get("beneficiary_type")),
            ("Category", intent_data.get("transaction_category")),
            ("Purpose", intent_data.get("purpose")),
        ]
        with self.body:
            for label, value in rows:
                if value in (None, ""):
                    continue
                with ui.row().classes("justify-between w-full"):
                    ui.label(label).classes("text-xs").style(f"color: {Theme.TEXT_MUTED}")
                    ui.label(str(value)).classes("text-sm avc-mono").style(
                        f"color: {Theme.TEXT_PRIMARY}"
                    )

    @staticmethod
    def _format_amount(intent_data: dict) -> Optional[str]:
        amount = intent_data.get("amount")
        currency = intent_data.get("currency")
        if amount is None:
            return None
        return f"{amount} {currency}" if currency else str(amount)


class RiskCard(CollapsibleCard):
    """Displays Risk Engine output: risk score, level, and triggered rules."""

    def __init__(self) -> None:
        super().__init__("Risk Assessment", "shield")

    def update(self, risk_data: Optional[dict]) -> None:
        self.clear()
        if not risk_data:
            self.hide()
            return
        self.show()
        level = (risk_data.get("risk_level") or "").upper()
        level_color = {
            "LOW": Theme.ACCENT_GREEN,
            "MEDIUM": Theme.ACCENT_AMBER,
            "HIGH": Theme.ACCENT_RED,
            "CRITICAL": Theme.ACCENT_RED,
        }.get(level, Theme.TEXT_SECONDARY)
        with self.body:
            if risk_data.get("risk_score") is not None:
                with ui.row().classes("justify-between w-full"):
                    ui.label("Risk Score").classes("text-xs").style(f"color: {Theme.TEXT_MUTED}")
                    ui.label(f"{risk_data['risk_score']:.2f}" if isinstance(
                        risk_data["risk_score"], (int, float)
                    ) else str(risk_data["risk_score"])).classes("text-sm avc-mono").style(
                        f"color: {level_color}"
                    )
            if level:
                with ui.row().classes("justify-between w-full items-center"):
                    ui.label("Risk Level").classes("text-xs").style(f"color: {Theme.TEXT_MUTED}")
                    ui.badge(level).style(
                        f"background-color: {level_color}33; color: {level_color};"
                    )
            reasons = risk_data.get("risk_reasons")
            if reasons:
                ui.separator().classes("my-1")
                for reason in reasons:
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("report", size="0.9rem").style(f"color: {Theme.ACCENT_AMBER}")
                        ui.label(str(reason)).classes("text-xs")


class PolicyCard(CollapsibleCard):
    """Displays Policy Engine output: matched rules and required action."""

    def __init__(self) -> None:
        super().__init__("Policy", "policy")

    def update(self, policy_data: Optional[dict]) -> None:
        self.clear()
        if not policy_data:
            self.hide()
            return
        self.show()
        with self.body:
            required_action = policy_data.get("required_action")
            if required_action:
                with ui.row().classes("justify-between w-full"):
                    ui.label("Required Action").classes("text-xs").style(
                        f"color: {Theme.TEXT_MUTED}"
                    )
                    ui.label(str(required_action)).classes("text-sm avc-mono").style(
                        f"color: {Theme.ACCENT_CYAN}"
                    )
            policy_reason = policy_data.get("policy_reason")
            if policy_reason:
                ui.label(str(policy_reason)).classes("text-xs").style(
                    f"color: {Theme.TEXT_SECONDARY}"
                )
            matched_rules = policy_data.get("matched_rules")
            if matched_rules:
                ui.separator().classes("my-1")
                ui.label("Matched Rules").classes("text-xs").style(f"color: {Theme.TEXT_MUTED}")
                with ui.row().classes("gap-1 flex-wrap"):
                    for rule in matched_rules:
                        ui.badge(str(rule)).style(
                            f"background-color: {Theme.ACCENT_VIOLET}33; color: {Theme.ACCENT_VIOLET};"
                        )


class FeatureAttentionCard(CollapsibleCard):
    """Horizontal Plotly bar chart of model feature-attention weights."""

    def __init__(self) -> None:
        super().__init__("Feature Attention", "bar_chart")
        self._plot: Optional[ui.plotly] = None

    def update(self, attention: Optional[dict]) -> None:
        self.clear()
        if not attention:
            self.hide()
            return
        self.show()
        names = list(attention.keys())
        values = [float(v) for v in attention.values()]
        order = sorted(range(len(names)), key=lambda i: values[i])
        names = [names[i] for i in order]
        values = [values[i] for i in order]
        fig = go.Figure(
            go.Bar(
                x=values,
                y=names,
                orientation="h",
                marker={"color": values, "colorscale": [[0, Theme.BG_PANEL], [1, Theme.ACCENT_CYAN]]},
            )
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": Theme.TEXT_PRIMARY, "size": 10, "family": Theme.FONT_MONO},
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            height=max(120, 26 * len(names)),
            xaxis={"gridcolor": Theme.BORDER},
            transition={"duration": 500},
        )
        with self.body:
            self._plot = ui.plotly(fig).classes("w-full")


class FeatureVectorCard(CollapsibleCard):
    """Collapsible, searchable table of the raw feature vector."""

    def __init__(self) -> None:
        super().__init__("Feature Vector", "table_rows")
        self._table: Optional[ui.table] = None
        self._search: Optional[ui.input] = None
        self._expansion: Optional[ui.expansion] = None

    def update(self, feature_vector: Optional[dict]) -> None:
        self.clear()
        if not feature_vector:
            self.hide()
            return
        self.show()
        rows = [{"feature": k, "value": str(v)} for k, v in feature_vector.items()]
        columns = [
            {"name": "feature", "label": "Feature", "field": "feature", "align": "left", "sortable": True},
            {"name": "value", "label": "Value", "field": "value", "align": "left", "sortable": True},
        ]
        with self.body:
            with ui.expansion("Show feature vector", icon="unfold_more").classes("w-full") as expansion:
                self._expansion = expansion
                self._search = ui.input(placeholder="Search features…").props("dense clearable").classes(
                    "w-full mb-1"
                )
                self._table = ui.table(columns=columns, rows=rows, row_key="feature").classes(
                    "w-full avc-mono text-xs"
                ).props("dense flat")
                self._search.bind_value(self._table, "filter")


class ReasonsTimelineCard(CollapsibleCard):
    """Vertical timeline of human-readable decision reasons/checks."""

    def __init__(self) -> None:
        super().__init__("Reasons", "checklist")

    def update(self, reasons: Optional[list], overall_ok: bool = True) -> None:
        self.clear()
        if not reasons:
            self.hide()
            return
        self.show()
        with self.body:
            for reason in reasons:
                icon = "check_circle" if overall_ok else "info"
                color = Theme.ACCENT_GREEN if overall_ok else Theme.ACCENT_CYAN
                with ui.row().classes("items-center gap-2 avc-fade-in"):
                    ui.icon(icon, size="1rem").style(f"color: {color}")
                    ui.label(str(reason)).classes("text-sm")


class JsonViewerCard(CollapsibleCard):
    """Tabbed pretty-printed JSON viewer for the raw request/response, with
    copy-to-clipboard and download support."""

    def __init__(self) -> None:
        super().__init__("Raw JSON", "data_object")
        self._request_area: Optional[ui.code] = None
        self._response_area: Optional[ui.code] = None
        self._request_text = "{}"
        self._response_text = "{}"
        self._build_tabs()

    def _build_tabs(self) -> None:
        with self.body:
            with ui.tabs().classes("w-full") as tabs:
                request_tab = ui.tab("Request")
                response_tab = ui.tab("Response")
            with ui.tab_panels(tabs, value=request_tab).classes("w-full"):
                with ui.tab_panel(request_tab):
                    with ui.row().classes("justify-end w-full gap-1 mb-1"):
                        ui.button(icon="content_copy", on_click=lambda: self._copy("request")).props(
                            "flat dense size=sm"
                        )
                        ui.button(icon="download", on_click=lambda: self._download("request")).props(
                            "flat dense size=sm"
                        )
                    self._request_area = ui.code("{}", language="json").classes(
                        "w-full avc-scrollbar"
                    ).style("max-height: 260px; overflow: auto;")
                with ui.tab_panel(response_tab):
                    with ui.row().classes("justify-end w-full gap-1 mb-1"):
                        ui.button(icon="content_copy", on_click=lambda: self._copy("response")).props(
                            "flat dense size=sm"
                        )
                        ui.button(icon="download", on_click=lambda: self._download("response")).props(
                            "flat dense size=sm"
                        )
                    self._response_area = ui.code("{}", language="json").classes(
                        "w-full avc-scrollbar"
                    ).style("max-height: 260px; overflow: auto;")

    def update(self, request_payload: dict, response_payload: dict) -> None:
        self._request_text = json.dumps(request_payload, indent=2, default=str)
        self._response_text = json.dumps(response_payload, indent=2, default=str)
        self._request_area.set_content(self._request_text)
        self._response_area.set_content(self._response_text)

    def _copy(self, which: str) -> None:
        text = self._request_text if which == "request" else self._response_text
        ui.run_javascript(f"navigator.clipboard.writeText({json.dumps(text)})")
        ui.notify(f"{which.capitalize()} JSON copied to clipboard", type="positive", position="top")

    def _download(self, which: str) -> None:
        text = self._request_text if which == "request" else self._response_text
        ui.download(text.encode("utf-8"), filename=f"authentication_{which}.json")


# =============================================================================
# Performance panel
# =============================================================================
class PerformancePanel(CollapsibleCard):
    """Displays per-stage execution times, total latency, and host telemetry."""

    STAGE_TIME_LABELS: dict[PipelineStage, str] = {
        PipelineStage.AUTH_NETWORK: "Authentication Network",
        PipelineStage.INTENT_ENGINE: "Intent Engine",
        PipelineStage.RISK_ENGINE: "Risk Engine",
        PipelineStage.POLICY_ENGINE: "Policy Engine",
        PipelineStage.DECISION_ENGINE: "Decision Engine",
    }

    def __init__(self, monitor: SystemMonitor) -> None:
        self._monitor = monitor
        super().__init__("Performance", "speed")
        self.show()  # always visible; total latency and host stats are always known

    def update(
        self,
        stage_times: dict[PipelineStage, Optional[float]],
        total_ms: float,
    ) -> None:
        self.clear()
        with self.body:
            for stage, label in self.STAGE_TIME_LABELS.items():
                duration = stage_times.get(stage)
                if duration is None:
                    continue  # hide gracefully when the backend didn't report this stage
                with ui.row().classes("justify-between w-full"):
                    ui.label(label).classes("text-xs").style(f"color: {Theme.TEXT_MUTED}")
                    ui.label(f"{duration:.1f} ms").classes("text-xs avc-mono").style(
                        f"color: {Theme.ACCENT_CYAN}"
                    )
            ui.separator().classes("my-1")
            with ui.row().classes("justify-between w-full"):
                ui.label("Total Latency").classes("text-xs font-bold").style(
                    f"color: {Theme.TEXT_SECONDARY}"
                )
                ui.label(f"{total_ms:.1f} ms").classes("text-xs avc-mono font-bold").style(
                    f"color: {Theme.ACCENT_GREEN}"
                )
            ui.separator().classes("my-1")
            cpu = self._monitor.cpu_percent()
            mem = self._monitor.memory_percent()
            gpu = self._monitor.gpu_percent()
            for label, value in (("CPU Usage", cpu), ("Memory Usage", mem), ("GPU Usage", gpu)):
                with ui.row().classes("justify-between w-full"):
                    ui.label(label).classes("text-xs").style(f"color: {Theme.TEXT_MUTED}")
                    text = f"{value:.0f}%" if value is not None else "N/A"
                    ui.label(text).classes("text-xs avc-mono").style(f"color: {Theme.TEXT_PRIMARY}")


# =============================================================================
# Top bar
# =============================================================================
class TopBar:
    """The header strip: title, live system telemetry, and clock."""

    def __init__(self, monitor: SystemMonitor) -> None:
        self._monitor = monitor
        self._cpu_label: Optional[ui.label] = None
        self._mem_label: Optional[ui.label] = None
        self._gpu_label: Optional[ui.label] = None
        self._latency_label: Optional[ui.label] = None
        self._clock_label: Optional[ui.label] = None
        self._build()
        ui.timer(2.0, self._refresh)

    def _stat(self, icon: str, initial: str, color: str) -> ui.label:
        with ui.row().classes("items-center gap-1"):
            ui.icon(icon, size="1rem").style(f"color: {color}")
            label = ui.label(initial).classes("avc-mono text-xs").style(f"color: {Theme.TEXT_PRIMARY}")
        return label

    def _build(self) -> None:
        with ui.header().classes("items-center justify-between px-6").style(
            f"background: {Theme.BG_BASE}; border-bottom: 1px solid {Theme.BORDER};"
        ):
            with ui.row().classes("items-center gap-3"):
                ui.icon("directions_car", size="1.8rem").style(f"color: {Theme.ACCENT_CYAN}")
                ui.label("VEHICLE AUTHENTICATION COMMAND CENTER").classes("avc-title text-base").style(
                    f"color: {Theme.TEXT_PRIMARY}; letter-spacing: 0.12em;"
                )
            with ui.row().classes("items-center gap-5"):
                self._gpu_label = self._stat("developer_board", "GPU: —", Theme.ACCENT_GREEN)
                self._cpu_label = self._stat("memory", "CPU: —", Theme.ACCENT_CYAN)
                self._mem_label = self._stat("sd_card", "MEM: —", Theme.ACCENT_VIOLET)
                self._latency_label = self._stat("bolt", "Latency: —", Theme.ACCENT_AMBER)
                self._clock_label = self._stat("schedule", "--:--:--", Theme.TEXT_SECONDARY)

    def _refresh(self) -> None:
        cpu = self._monitor.cpu_percent()
        mem = self._monitor.memory_percent()
        gpu = self._monitor.gpu_percent()
        self._cpu_label.set_text(f"CPU: {cpu:.0f}%" if cpu is not None else "CPU: N/A")
        self._mem_label.set_text(f"MEM: {mem:.0f}%" if mem is not None else "MEM: N/A")
        self._gpu_label.set_text(f"GPU: {gpu:.0f}%" if gpu is not None else "GPU: N/A")
        self._clock_label.set_text(datetime.now().strftime("%H:%M:%S"))

    def set_latency(self, latency_ms: Optional[float]) -> None:
        """Update the header's total-latency indicator after a request completes."""
        self._latency_label.set_text(
            f"Latency: {latency_ms:.0f} ms" if latency_ms is not None else "Latency: —"
        )


# =============================================================================
# Left panel: authentication inputs
# =============================================================================
class InputPanel:
    """Renders the authentication request form and builds the outgoing
    payload matching the backend's `AuthenticationRequest` schema.

    Beyond the headline fields (transcript, user/vehicle IDs, GPS, speed,
    engine state), this panel also exposes every field that
    `engines.feature_extractor.FeatureExtractor.extract()` actually reads
    off of `identity` / `biometric` / `behavior` / `vehicle` / `history` /
    `transaction`. Those are the values that drive the Authentication
    Network's trust/risk/confidence scores -- omitting them (as this panel
    used to) meant every request silently fell back to the same defaults
    and the model's output never appeared to change.
    """

    TRANSACTION_CATEGORIES: tuple[str, ...] = ("TRANSFER", "UPI", "BILL", "SHOPPING", "ATM", "OTHER")
    BENEFICIARY_TYPES: tuple[str, ...] = ("SAVED", "NEW", "SELF", "MERCHANT", "OTHER")
    INTENT_TYPES: tuple[str, ...] = (
        "UNKNOWN", "BALANCE_INQUIRY", "MONEY_TRANSFER", "BILL_PAYMENT", "TRANSACTION_HISTORY",
    )

    EXAMPLE_PAYLOAD: dict[str, Any] = {
        "transcript": "Hey, please send fifteen thousand rupees to my brother Raj for rent.",
        "user_id": "user_8842",
        "vehicle_id": "VH-TESLA-3311",
        "gps_lat": 28.6139,
        "gps_lon": 77.2090,
        "speed": 42.0,
        "engine_running": True,
        # -- identity --
        "account_age_days": 640,
        "kyc_verified": True,
        "phone_verified": True,
        "email_verified": True,
        "voice_enrolled": True,
        # -- biometric --
        "speaker_similarity": 0.91,
        "liveness_score": 0.95,
        "audio_quality": 0.88,
        "spoof_probability": 0.03,
        # -- behavior --
        "speech_rate_similarity": 0.86,
        "pronunciation_similarity": 0.9,
        "command_familiarity": 0.8,
        "stress_score": 0.15,
        "hesitation_score": 0.1,
        # -- vehicle context --
        "location_familiarity": 0.9,
        "time_familiarity": 0.8,
        "driver_present": True,
        "seatbelt_fastened": True,
        # -- history --
        "previous_trust_score": 0.87,
        "failed_attempts": 0,
        "successful_transactions": 128,
        "fraud_history": False,
        # -- transaction --
        "amount": 15000.0,
        "category": "TRANSFER",
        "beneficiary_type": "NEW",
        "beneficiary_frequency": 0.0,
        "intent": "MONEY_TRANSFER",
        "llm_confidence": 0.92,
        "transaction_risk": 0.35,
    }

    def __init__(self, on_authenticate: Callable[[], None]) -> None:
        self._on_authenticate = on_authenticate
        self.transcript_input: Optional[ui.textarea] = None
        self.user_id_input: Optional[ui.input] = None
        self.vehicle_id_input: Optional[ui.input] = None
        self.gps_lat_input: Optional[ui.number] = None
        self.gps_lon_input: Optional[ui.number] = None
        self.speed_slider: Optional[ui.slider] = None
        self.speed_display: Optional[ui.label] = None
        self.engine_toggle: Optional[ui.switch] = None
        self.timestamp_input: Optional[ui.input] = None
        self.authenticate_button: Optional[ui.button] = None

        # -- identity & verification --
        self.account_age_input: Optional[ui.number] = None
        self.kyc_toggle: Optional[ui.switch] = None
        self.phone_verified_toggle: Optional[ui.switch] = None
        self.email_verified_toggle: Optional[ui.switch] = None
        self.voice_enrolled_toggle: Optional[ui.switch] = None

        # -- voice biometrics --
        self.speaker_similarity_slider: Optional[ui.slider] = None
        self.liveness_slider: Optional[ui.slider] = None
        self.audio_quality_slider: Optional[ui.slider] = None
        self.spoof_probability_slider: Optional[ui.slider] = None

        # -- behavioral signals --
        self.speech_rate_slider: Optional[ui.slider] = None
        self.pronunciation_slider: Optional[ui.slider] = None
        self.command_familiarity_slider: Optional[ui.slider] = None
        self.stress_slider: Optional[ui.slider] = None
        self.hesitation_slider: Optional[ui.slider] = None

        # -- vehicle context (advanced) --
        self.location_familiarity_slider: Optional[ui.slider] = None
        self.time_familiarity_slider: Optional[ui.slider] = None
        self.driver_present_toggle: Optional[ui.switch] = None
        self.seatbelt_toggle: Optional[ui.switch] = None

        # -- account history --
        self.previous_trust_slider: Optional[ui.slider] = None
        self.failed_attempts_input: Optional[ui.number] = None
        self.successful_transactions_input: Optional[ui.number] = None
        self.fraud_history_toggle: Optional[ui.switch] = None

        # -- transaction details --
        self.amount_input: Optional[ui.number] = None
        self.category_select: Optional[ui.select] = None
        self.beneficiary_type_select: Optional[ui.select] = None
        self.beneficiary_frequency_input: Optional[ui.number] = None
        self.intent_select: Optional[ui.select] = None
        self.llm_confidence_slider: Optional[ui.slider] = None
        self.transaction_risk_slider: Optional[ui.slider] = None

        self._build()

    def _fraction_slider(
        self, label: str, default: float = 0.5
    ) -> ui.slider:
        """A labeled 0-1 slider with a live percentage readout, matching
        the visual pattern already used for the speed slider."""
        with ui.column().classes("w-full gap-0"):
            with ui.row().classes("justify-between w-full"):
                ui.label(label).classes("text-xs").style(f"color: {Theme.TEXT_MUTED}")
                display = ui.label(f"{default:.2f}").classes("text-xs avc-mono").style(
                    f"color: {Theme.ACCENT_CYAN}"
                )
            slider = ui.slider(min=0.0, max=1.0, value=default, step=0.01)
            slider.on("update:model-value", lambda e: display.set_text(f"{e.args:.2f}"))
            slider._display_label = display  # noqa: SLF001 - stash for reset/example loading
        return slider

    def _set_fraction_slider(self, slider: ui.slider, value: float) -> None:
        slider.set_value(value)
        slider._display_label.set_text(f"{value:.2f}")  # noqa: SLF001

    def _build(self) -> None:
        with ui.column().classes("avc-card w-full gap-3"):
            ui.label("AUTHENTICATION REQUEST").classes("avc-title text-sm").style(
                f"color: {Theme.TEXT_SECONDARY}; letter-spacing: 0.15em;"
            )

            self.transcript_input = ui.textarea(
                label="Transcript", placeholder="What the occupant said to the vehicle assistant…"
            ).classes("w-full").props("outlined dense rows=4")

            with ui.row().classes("w-full gap-2"):
                self.user_id_input = ui.input(label="User ID").classes("flex-1").props("outlined dense")
                self.vehicle_id_input = ui.input(label="Vehicle ID").classes("flex-1").props(
                    "outlined dense"
                )

            with ui.row().classes("w-full gap-2"):
                self.gps_lat_input = ui.number(label="GPS Latitude", format="%.4f").classes(
                    "flex-1"
                ).props("outlined dense")
                self.gps_lon_input = ui.number(label="GPS Longitude", format="%.4f").classes(
                    "flex-1"
                ).props("outlined dense")

            with ui.column().classes("w-full gap-0"):
                with ui.row().classes("justify-between w-full"):
                    ui.label("Speed").classes("text-xs").style(f"color: {Theme.TEXT_MUTED}")
                    self.speed_display = ui.label("0 km/h").classes("text-xs avc-mono").style(
                        f"color: {Theme.ACCENT_CYAN}"
                    )
                self.speed_slider = ui.slider(min=0, max=200, value=0, step=1).props("label-always")
                self.speed_slider.on(
                    "update:model-value",
                    lambda e: self.speed_display.set_text(f"{e.args:.0f} km/h"),
                )

            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Engine Running").classes("text-sm").style(f"color: {Theme.TEXT_SECONDARY}")
                self.engine_toggle = ui.switch(value=True)

            self.timestamp_input = ui.input(
                label="Timestamp (ISO 8601)",
                value=datetime.now().isoformat(timespec="seconds"),
            ).classes("w-full").props("outlined dense")

            ui.separator().classes("my-1")

            # -- Identity & Verification -----------------------------------
            with ui.expansion("Identity & Verification", icon="badge").classes("w-full"):
                with ui.column().classes("w-full gap-2 pt-1"):
                    self.account_age_input = ui.number(
                        label="Account Age (days)", value=0, min=0
                    ).classes("w-full").props("outlined dense")
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label("KYC Verified").classes("text-sm").style(f"color: {Theme.TEXT_SECONDARY}")
                        self.kyc_toggle = ui.switch(value=False)
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label("Phone Verified").classes("text-sm").style(f"color: {Theme.TEXT_SECONDARY}")
                        self.phone_verified_toggle = ui.switch(value=False)
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label("Email Verified").classes("text-sm").style(f"color: {Theme.TEXT_SECONDARY}")
                        self.email_verified_toggle = ui.switch(value=False)
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label("Voice Enrolled").classes("text-sm").style(f"color: {Theme.TEXT_SECONDARY}")
                        self.voice_enrolled_toggle = ui.switch(value=False)

            # -- Voice Biometrics --------------------------------------------
            with ui.expansion("Voice Biometrics", icon="graphic_eq").classes("w-full"):
                with ui.column().classes("w-full gap-2 pt-1"):
                    self.speaker_similarity_slider = self._fraction_slider("Speaker Similarity", 0.5)
                    self.liveness_slider = self._fraction_slider("Liveness Score", 0.5)
                    self.audio_quality_slider = self._fraction_slider("Audio Quality", 0.5)
                    self.spoof_probability_slider = self._fraction_slider("Spoof Probability", 0.1)

            # -- Behavioral Signals --------------------------------------------
            with ui.expansion("Behavioral Signals", icon="psychology").classes("w-full"):
                with ui.column().classes("w-full gap-2 pt-1"):
                    self.speech_rate_slider = self._fraction_slider("Speech Rate Similarity", 0.5)
                    self.pronunciation_slider = self._fraction_slider("Pronunciation Similarity", 0.5)
                    self.command_familiarity_slider = self._fraction_slider("Command Familiarity", 0.5)
                    self.stress_slider = self._fraction_slider("Stress Score", 0.2)
                    self.hesitation_slider = self._fraction_slider("Hesitation Score", 0.2)

            # -- Vehicle Context (advanced) -----------------------------------
            with ui.expansion("Vehicle Context (Advanced)", icon="directions_car").classes("w-full"):
                with ui.column().classes("w-full gap-2 pt-1"):
                    self.location_familiarity_slider = self._fraction_slider("Location Familiarity", 0.5)
                    self.time_familiarity_slider = self._fraction_slider("Time Familiarity", 0.5)
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label("Driver Present").classes("text-sm").style(f"color: {Theme.TEXT_SECONDARY}")
                        self.driver_present_toggle = ui.switch(value=True)
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label("Seatbelt Fastened").classes("text-sm").style(
                            f"color: {Theme.TEXT_SECONDARY}"
                        )
                        self.seatbelt_toggle = ui.switch(value=True)

            # -- Account History -----------------------------------------------
            with ui.expansion("Account History", icon="history").classes("w-full"):
                with ui.column().classes("w-full gap-2 pt-1"):
                    self.previous_trust_slider = self._fraction_slider("Previous Trust Score", 1.0)
                    self.failed_attempts_input = ui.number(
                        label="Failed Attempts", value=0, min=0
                    ).classes("w-full").props("outlined dense")
                    self.successful_transactions_input = ui.number(
                        label="Successful Transactions", value=0, min=0
                    ).classes("w-full").props("outlined dense")
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label("Fraud History").classes("text-sm").style(f"color: {Theme.TEXT_SECONDARY}")
                        self.fraud_history_toggle = ui.switch(value=False)

            # -- Transaction Details ---------------------------------------------
            with ui.expansion("Transaction Details", icon="payments", value=True).classes("w-full"):
                with ui.column().classes("w-full gap-2 pt-1"):
                    self.amount_input = ui.number(
                        label="Amount", value=0, min=0
                    ).classes("w-full").props("outlined dense")
                    self.category_select = ui.select(
                        list(self.TRANSACTION_CATEGORIES), label="Category", value="OTHER"
                    ).classes("w-full").props("outlined dense")
                    self.beneficiary_type_select = ui.select(
                        list(self.BENEFICIARY_TYPES), label="Beneficiary Type", value="OTHER"
                    ).classes("w-full").props("outlined dense")
                    self.beneficiary_frequency_input = ui.number(
                        label="Beneficiary Frequency (past transfers)", value=0, min=0
                    ).classes("w-full").props("outlined dense")
                    self.intent_select = ui.select(
                        list(self.INTENT_TYPES), label="Intent", value="UNKNOWN"
                    ).classes("w-full").props("outlined dense")
                    self.llm_confidence_slider = self._fraction_slider("LLM Confidence", 0.5)
                    self.transaction_risk_slider = self._fraction_slider("Transaction Risk", 0.1)

            ui.separator().classes("my-1")

            with ui.row().classes("w-full gap-2"):
                self.authenticate_button = ui.button(
                    "AUTHENTICATE", icon="bolt", on_click=self._on_authenticate
                ).classes("flex-1 avc-glow-cyan").props(f'color="cyan-9" unelevated')
            with ui.row().classes("w-full gap-2"):
                ui.button("Load Example", icon="science", on_click=self.load_example).classes(
                    "flex-1"
                ).props("outline")
                ui.button("Clear", icon="delete_sweep", on_click=self.clear).classes("flex-1").props(
                    "outline color=grey"
                )

    def load_example(self) -> None:
        """Populate every input with a realistic example authentication request."""
        example = self.EXAMPLE_PAYLOAD
        self.transcript_input.set_value(example["transcript"])
        self.user_id_input.set_value(example["user_id"])
        self.vehicle_id_input.set_value(example["vehicle_id"])
        self.gps_lat_input.set_value(example["gps_lat"])
        self.gps_lon_input.set_value(example["gps_lon"])
        self.speed_slider.set_value(example["speed"])
        self.speed_display.set_text(f"{example['speed']:.0f} km/h")
        self.engine_toggle.set_value(example["engine_running"])
        self.timestamp_input.set_value(datetime.now().isoformat(timespec="seconds"))

        self.account_age_input.set_value(example["account_age_days"])
        self.kyc_toggle.set_value(example["kyc_verified"])
        self.phone_verified_toggle.set_value(example["phone_verified"])
        self.email_verified_toggle.set_value(example["email_verified"])
        self.voice_enrolled_toggle.set_value(example["voice_enrolled"])

        self._set_fraction_slider(self.speaker_similarity_slider, example["speaker_similarity"])
        self._set_fraction_slider(self.liveness_slider, example["liveness_score"])
        self._set_fraction_slider(self.audio_quality_slider, example["audio_quality"])
        self._set_fraction_slider(self.spoof_probability_slider, example["spoof_probability"])

        self._set_fraction_slider(self.speech_rate_slider, example["speech_rate_similarity"])
        self._set_fraction_slider(self.pronunciation_slider, example["pronunciation_similarity"])
        self._set_fraction_slider(self.command_familiarity_slider, example["command_familiarity"])
        self._set_fraction_slider(self.stress_slider, example["stress_score"])
        self._set_fraction_slider(self.hesitation_slider, example["hesitation_score"])

        self._set_fraction_slider(self.location_familiarity_slider, example["location_familiarity"])
        self._set_fraction_slider(self.time_familiarity_slider, example["time_familiarity"])
        self.driver_present_toggle.set_value(example["driver_present"])
        self.seatbelt_toggle.set_value(example["seatbelt_fastened"])

        self._set_fraction_slider(self.previous_trust_slider, example["previous_trust_score"])
        self.failed_attempts_input.set_value(example["failed_attempts"])
        self.successful_transactions_input.set_value(example["successful_transactions"])
        self.fraud_history_toggle.set_value(example["fraud_history"])

        self.amount_input.set_value(example["amount"])
        self.category_select.set_value(example["category"])
        self.beneficiary_type_select.set_value(example["beneficiary_type"])
        self.beneficiary_frequency_input.set_value(example["beneficiary_frequency"])
        self.intent_select.set_value(example["intent"])
        self._set_fraction_slider(self.llm_confidence_slider, example["llm_confidence"])
        self._set_fraction_slider(self.transaction_risk_slider, example["transaction_risk"])

        ui.notify("Example request loaded", type="info", position="top")

    def clear(self) -> None:
        """Reset every input field to its blank/default state."""
        self.transcript_input.set_value("")
        self.user_id_input.set_value("")
        self.vehicle_id_input.set_value("")
        self.gps_lat_input.set_value(None)
        self.gps_lon_input.set_value(None)
        self.speed_slider.set_value(0)
        self.speed_display.set_text("0 km/h")
        self.engine_toggle.set_value(False)
        self.timestamp_input.set_value(datetime.now().isoformat(timespec="seconds"))

        self.account_age_input.set_value(0)
        self.kyc_toggle.set_value(False)
        self.phone_verified_toggle.set_value(False)
        self.email_verified_toggle.set_value(False)
        self.voice_enrolled_toggle.set_value(False)

        for slider, default in (
            (self.speaker_similarity_slider, 0.5),
            (self.liveness_slider, 0.5),
            (self.audio_quality_slider, 0.5),
            (self.spoof_probability_slider, 0.1),
            (self.speech_rate_slider, 0.5),
            (self.pronunciation_slider, 0.5),
            (self.command_familiarity_slider, 0.5),
            (self.stress_slider, 0.2),
            (self.hesitation_slider, 0.2),
            (self.location_familiarity_slider, 0.5),
            (self.time_familiarity_slider, 0.5),
            (self.previous_trust_slider, 1.0),
            (self.llm_confidence_slider, 0.5),
            (self.transaction_risk_slider, 0.1),
        ):
            self._set_fraction_slider(slider, default)

        self.driver_present_toggle.set_value(True)
        self.seatbelt_toggle.set_value(True)

        self.failed_attempts_input.set_value(0)
        self.successful_transactions_input.set_value(0)
        self.fraud_history_toggle.set_value(False)

        self.amount_input.set_value(0)
        self.category_select.set_value("OTHER")
        self.beneficiary_type_select.set_value("OTHER")
        self.beneficiary_frequency_input.set_value(0)
        self.intent_select.set_value("UNKNOWN")

    def build_payload(self) -> dict[str, Any]:
        return {
            # ===========================
            # Required TransactionRequest fields
            # ===========================
            "user_id": self.user_id_input.value or "",
            "transcript": self.transcript_input.value or "",
            "audio_path": None,

            "vehicle_id": self.vehicle_id_input.value or "",

            "gps_latitude": self.gps_lat_input.value,
            "gps_longitude": self.gps_lon_input.value,

            "vehicle_speed": self.speed_slider.value,
            "engine_running": self.engine_toggle.value,

            "timestamp": self.timestamp_input.value,

            "session_id": None,
            "metadata": {},

            # ===========================
            # FeatureExtractor input
            #
            # These keys are read verbatim by
            # `engines.feature_extractor.FeatureExtractor.extract()` --
            # they must match its lookups exactly or the corresponding
            # feature silently falls back to a constant default and the
            # Authentication Network's score stops responding to it.
            # ===========================
            "identity": {
                "user_id": self.user_id_input.value or "",
                "account_age_days": self.account_age_input.value or 0,
                "kyc_verified": self.kyc_toggle.value,
                "phone_verified": self.phone_verified_toggle.value,
                "email_verified": self.email_verified_toggle.value,
                "voice_enrolled": self.voice_enrolled_toggle.value,
            },

            "biometric": {
                "speaker_similarity": self.speaker_similarity_slider.value,
                "liveness_score": self.liveness_slider.value,
                "audio_quality": self.audio_quality_slider.value,
                "spoof_probability": self.spoof_probability_slider.value,
            },

            "behavior": {
                "speech_rate_similarity": self.speech_rate_slider.value,
                "pronunciation_similarity": self.pronunciation_slider.value,
                "command_familiarity": self.command_familiarity_slider.value,
                "stress_score": self.stress_slider.value,
                "hesitation_score": self.hesitation_slider.value,
            },

            "vehicle": {
                "vehicle_id": self.vehicle_id_input.value or "",
                "vehicle_speed": self.speed_slider.value,
                "engine_running": self.engine_toggle.value,
                "location_familiarity": self.location_familiarity_slider.value,
                "time_familiarity": self.time_familiarity_slider.value,
                "driver_present": self.driver_present_toggle.value,
                "seatbelt_fastened": self.seatbelt_toggle.value,
                "gps": {
                    "lat": self.gps_lat_input.value,
                    "lon": self.gps_lon_input.value,
                },
            },

            "history": {
                "previous_trust_score": self.previous_trust_slider.value,
                "failed_attempts": self.failed_attempts_input.value or 0,
                "successful_transactions": self.successful_transactions_input.value or 0,
                "fraud_history": self.fraud_history_toggle.value,
            },

            "transaction": {
                "amount": self.amount_input.value or 0.0,
                "category": self.category_select.value,
                "beneficiary_type": self.beneficiary_type_select.value,
                "beneficiary_frequency": self.beneficiary_frequency_input.value or 0.0,
                "intent": self.intent_select.value,
                "llm_confidence": self.llm_confidence_slider.value,
                "transaction_risk": self.transaction_risk_slider.value,
            },
        }


# =============================================================================
# Main dashboard orchestrator
# =============================================================================
class AuthDashboard:
    """Top-level orchestrator that assembles the page layout and wires the
    input panel, pipeline visualizer, and result cards to the backend API.

    This is the single class that `app.py` instantiates and builds.
    """

    STAGE_TIMING_KEYWORDS: dict[PipelineStage, tuple[str, ...]] = {
        PipelineStage.FEATURE_EXTRACTION: ("feature_extract", "feature_time", "extraction_time"),
        PipelineStage.AUTH_NETWORK: ("auth_network", "authentication_network", "network_time"),
        PipelineStage.INTENT_ENGINE: ("intent_time", "intent_engine"),
        PipelineStage.RISK_ENGINE: ("risk_time", "risk_engine"),
        PipelineStage.POLICY_ENGINE: ("policy_time", "policy_engine"),
        PipelineStage.DECISION_ENGINE: ("decision_time", "decision_engine"),
    }

    def __init__(self, backend_url: str) -> None:
        self._api_client = APIClient(backend_url)
        self._monitor = SystemMonitor()

        # Components are constructed lazily inside `build()`'s `@ui.page`
        # so that a fresh set is created per browser session/connection.
        self.top_bar: Optional[TopBar] = None
        self.input_panel: Optional[InputPanel] = None
        self.pipeline: Optional[PipelineVisualizer] = None
        self.decision_badge: Optional[DecisionBadge] = None
        self.trust_gauge: Optional[ScoreGauge] = None
        self.risk_gauge: Optional[ScoreGauge] = None
        self.confidence_gauge: Optional[ScoreGauge] = None
        self.intent_card: Optional[IntentCard] = None
        self.risk_card: Optional[RiskCard] = None
        self.policy_card: Optional[PolicyCard] = None
        self.feature_attention_card: Optional[FeatureAttentionCard] = None
        self.feature_vector_card: Optional[FeatureVectorCard] = None
        self.reasons_card: Optional[ReasonsTimelineCard] = None
        self.json_viewer: Optional[JsonViewerCard] = None
        self.performance_panel: Optional[PerformancePanel] = None

    # -- Layout construction ------------------------------------------------
    def build(self) -> None:
        """Register the NiceGUI page route and construct the full layout."""

        @ui.page("/")
        def _page() -> None:
            Theme.inject_global_styles()
            self.top_bar = TopBar(self._monitor)

            with ui.row().classes("w-full gap-4 p-4 flex-nowrap items-start no-wrap").style(
                "min-height: calc(100vh - 64px);"
            ):
                # Left panel -------------------------------------------------
                with ui.column().classes("gap-4").style("width: 340px; flex-shrink: 0;"):
                    self.input_panel = InputPanel(on_authenticate=self._handle_authenticate)

                # Center panel -------------------------------------------------
                with ui.column().classes("gap-4 flex-1 min-w-0"):
                    self.pipeline = PipelineVisualizer()

                    with ui.column().classes("avc-card w-full gap-3"):
                        ui.label("SCORES").classes("avc-title text-sm").style(
                            f"color: {Theme.TEXT_SECONDARY}; letter-spacing: 0.15em;"
                        )
                        with ui.row().classes("w-full gap-2"):
                            self.trust_gauge = ScoreGauge("TRUST SCORE", Theme.ACCENT_GREEN)
                            self.risk_gauge = ScoreGauge("RISK SCORE", Theme.ACCENT_RED)
                            self.confidence_gauge = ScoreGauge("CONFIDENCE", Theme.ACCENT_CYAN)

                    self.feature_attention_card = FeatureAttentionCard()
                    self.feature_vector_card = FeatureVectorCard()
                    self.json_viewer = JsonViewerCard()

                # Right panel -------------------------------------------------
                with ui.column().classes("gap-4").style(
                    "width: 360px; flex-shrink: 0; max-height: calc(100vh - 96px); "
                    "overflow-y: auto;"
                ).classes("avc-scrollbar"):
                    self.decision_badge = DecisionBadge()
                    self.intent_card = IntentCard()
                    self.risk_card = RiskCard()
                    self.policy_card = PolicyCard()
                    self.reasons_card = ReasonsTimelineCard()
                    self.performance_panel = PerformancePanel(self._monitor)

            # All optional widgets start hidden until real data arrives.
            self._reset_result_widgets()

    # -- Request lifecycle ----------------------------------------------------
    def _reset_result_widgets(self) -> None:
        """Hide every optional result widget prior to the first request."""
        self.pipeline.reset()
        self.trust_gauge.hide()
        self.risk_gauge.hide()
        self.confidence_gauge.hide()
        self.intent_card.hide()
        self.risk_card.hide()
        self.policy_card.hide()
        self.feature_attention_card.hide()
        self.feature_vector_card.hide()
        self.reasons_card.hide()

    async def _handle_authenticate(self) -> None:
        """Handle the Authenticate button click end-to-end.

        Runs the pipeline animation and the real backend request
        concurrently, then populates every result widget -- hiding any
        whose backing field the backend did not return.
        """
        self.input_panel.authenticate_button.props("loading")
        self.input_panel.authenticate_button.disable()
        self.pipeline.reset()
        self.decision_badge.set_pending()

        payload = self.input_panel.build_payload()

        try:
            animation_task = asyncio.create_task(self._animate_placeholder_pipeline())
            request_task = asyncio.create_task(self._api_client.authenticate(payload))
            _, result = await asyncio.gather(animation_task, request_task)

            if not result.ok:
                self.pipeline.set_error_at(PipelineStage.FEATURE_EXTRACTION)
                self.decision_badge.set_error(result.error or "Unknown error contacting backend.")
                self.top_bar.set_latency(result.latency_ms)
                self.json_viewer.update(payload, {"error": result.error})
                ui.notify(result.error or "Authentication request failed.", type="negative")
                return

            self._populate_results(payload, result.response, result.latency_ms)

        finally:
            self.input_panel.authenticate_button.props(remove="loading")
            self.input_panel.authenticate_button.enable()

    async def _animate_placeholder_pipeline(self, step_delay: float = 0.16) -> None:
        """Light up each pipeline stage in sequence to visualize progress
        while the real backend request is in flight, per the required
        animation flow. Real per-stage timings (if the backend supplies
        them) are applied afterward in `_populate_results`."""
        empty_times: dict[PipelineStage, Optional[float]] = {stage: None for stage in PipelineVisualizer.STAGES}
        await self.pipeline.animate_pass(empty_times, step_delay=step_delay)

    def _extract_stage_times(self, response: dict) -> dict[PipelineStage, Optional[float]]:
        """Best-effort extraction of per-stage execution times from the
        response's `audit_log` (or any nested dict), matching the widget
        to `None` -- and therefore hiding it -- when the backend doesn't
        report that particular stage's timing."""
        audit_log = response.get("audit_log") or {}
        return {
            stage: find_numeric_by_keywords(audit_log, keywords)
            for stage, keywords in self.STAGE_TIMING_KEYWORDS.items()
        }

    def _populate_results(
        self, request_payload: dict, response: dict, client_latency_ms: float
    ) -> None:
        """Populate every dashboard widget from a successful backend response."""
        audit_log = response.get("audit_log") or {}

        # Re-run the pipeline nodes into their "done" state with any real
        # per-stage timings the backend provided (or a checkmark otherwise).
        stage_times = self._extract_stage_times(response)
        for stage in PipelineVisualizer.STAGES:
            self.pipeline.nodes[stage].set_done(stage_times.get(stage))
        self.pipeline.set_total(client_latency_ms)
        self.top_bar.set_latency(client_latency_ms)

        # Decision badge -------------------------------------------------
        status = response.get("status") or response.get("action")
        message = response.get("message") or response.get("reason")
        self.decision_badge.set_result(status, message)

        # Scores (Authentication Network) --------------------------------
        network_data = find_dict_by_keywords(audit_log, ("authentication_network", "auth_network"))
        trust_score = safe_get(network_data, "trust_score") if network_data else find_numeric_by_keywords(
            audit_log, ("trust_score",)
        )
        risk_score = find_numeric_by_keywords(audit_log, ("risk_score",))
        confidence = safe_get(network_data, "confidence") if network_data else find_numeric_by_keywords(
            audit_log, ("confidence",)
        )
        self.trust_gauge.update(trust_score)
        self.risk_gauge.update(risk_score)
        self.confidence_gauge.update(confidence)

        # Intent card ------------------------------------------------------
        intent_data = find_dict_by_keywords(audit_log, ("intent",))
        self.intent_card.update(intent_data)

        # Risk card ----------------------------------------------------------
        risk_data = find_dict_by_keywords(audit_log, ("risk",))
        self.risk_card.update(risk_data)

        # Policy card --------------------------------------------------------
        policy_data = find_dict_by_keywords(audit_log, ("policy",))
        self.policy_card.update(policy_data)

        # Feature attention & feature vector -----------------------------
        attention = find_dict_by_keywords(audit_log, ("attention", "feature_weight", "feature_importance"))
        self.feature_attention_card.update(attention)

        feature_vector = find_dict_by_keywords(audit_log, ("feature_vector", "features"))
        self.feature_vector_card.update(feature_vector)

        # Reasons timeline -----------------------------------------------
        reasons = find_list_by_keywords(audit_log, ("reasons", "checks", "verifications"))
        allowed = bool(response.get("transaction_allowed", True))
        self.reasons_card.update(reasons, overall_ok=allowed)

        # Performance panel ------------------------------------------------
        self.performance_panel.update(stage_times, client_latency_ms)

        # JSON viewer --------------------------------------------------------
        self.json_viewer.update(request_payload, response)