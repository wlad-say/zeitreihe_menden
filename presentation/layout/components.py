"""
layout/components.py
Wiederverwendbare UI-Bausteine für alle Tabs.
Wissenschaftliches Design — kein Browser-Außenrahmen.
"""

import io, base64
import matplotlib.pyplot as plt
from dash import dcc, html


# ── Farbpalette ──────────────────────────────────────────────────────────────

COLORS = {
    "primary":      "#1a2535",
    "accent":       "#2e5f8a",
    "accent_light": "#d6e4f0",
    "background":   "#eef0f3",
    "surface":      "#f5f6f8",
    "border":       "#cdd3da",
    "text":         "#1e2b3a",
    "subtle":       "#5a6878",
    "rule":         "#b0bac5",
}

# ── Layout-Styles ─────────────────────────────────────────────────────────────

STYLE_PAGE = {
    "fontFamily": "'Segoe UI', Arial, sans-serif",
    "background":  "#eef0f3",
    "margin":      "0",
    "padding":     "0",
    "minHeight":   "100vh",
}

STYLE_CARD = {
    "background":   "#f5f6f8",
    "border":       "1px solid #cdd3da",
    "borderRadius": "4px",
    "padding":      "20px 28px 24px 28px",
    "marginBottom": "16px",
}

STYLE_TABS = {
    "borderBottom": "2px solid #cdd3da",
    "background":   "#eef0f3",
    "margin":       "0",
    "padding":      "0 28px",
}

STYLE_TAB = {
    "padding":       "10px 22px",
    "fontWeight":    "400",
    "fontFamily":    "'Segoe UI', Arial, sans-serif",
    "fontSize":      "0.88rem",
    "letterSpacing": "0.03em",
    "color":         "#5a6878",
    "background":    "transparent",
    "border":        "none",
}

STYLE_TAB_SELECTED = {
    "padding":       "10px 22px",
    "fontWeight":    "600",
    "fontFamily":    "'Segoe UI', Arial, sans-serif",
    "fontSize":      "0.88rem",
    "letterSpacing": "0.03em",
    "color":         "#2e5f8a",
    "background":    "#f5f6f8",
    "borderTop":     "2px solid #2e5f8a",
    "borderBottom":  "none",
    "borderLeft":    "1px solid #cdd3da",
    "borderRight":   "1px solid #cdd3da",
}

STYLE_CONTENT = {
    "maxWidth": "1200px",
    "margin":   "0 auto",
    "padding":  "24px 28px 50px 28px",
}


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


# ── Komponenten ──────────────────────────────────────────────────────────────

def plot_image(b64: str, style: dict = None) -> html.Img:
    s = {
        "width":        "100%",
        "borderRadius": "3px",
        "border":       "1px solid #cdd3da",
        "marginBottom": "20px",
        "display":      "block",
    }
    if style:
        s.update(style)
    return html.Img(src=b64, style=s)


def info_block(text: str) -> html.Div:
    return html.Div(
        dcc.Markdown(text),
        style={
            "background":   "#d6e4f0",
            "padding":      "12px 16px",
            "borderLeft":   "3px solid #2e5f8a",
            "borderRadius": "2px",
            "marginBottom": "18px",
            "fontSize":     "0.875rem",
            "lineHeight":   "1.65",
            "color":        "#1e2b3a",
        },
    )


def section_title(text: str) -> html.Div:
    return html.Div([
        html.H4(
            text,
            style={
                "color":         "#1a2535",
                "margin":        "0 0 4px 0",
                "fontWeight":    "600",
                "fontSize":      "0.93rem",
                "letterSpacing": "0.02em",
                "fontFamily":    "'Segoe UI', Arial, sans-serif",
            },
        ),
        html.Hr(style={
            "border":    "none",
            "borderTop": "1px solid #b0bac5",
            "margin":    "0 0 14px 0",
        }),
    ])


def card(*children) -> html.Div:
    return html.Div(style=STYLE_CARD, children=list(children))


def results_table(rows: list[tuple]) -> html.Table:
    return html.Table(
        [
            html.Tr([
                html.Td(
                    k,
                    style={
                        "padding":    "5px 28px 5px 8px",
                        "color":      "#5a6878",
                        "fontWeight": "400",
                        "fontSize":   "0.85rem",
                        "whiteSpace": "nowrap",
                    },
                ),
                html.Td(
                    v,
                    style={
                        "padding":    "5px 8px",
                        "fontWeight": "600",
                        "fontSize":   "0.85rem",
                        "color":      "#1a2535",
                    },
                ),
            ])
            for k, v in rows
        ],
        style={
            "borderCollapse": "collapse",
            "marginBottom":   "16px",
            "borderTop":      "1px solid #cdd3da",
            "borderBottom":   "1px solid #cdd3da",
            "width":          "100%",
        },
    )