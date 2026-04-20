"""
Shared chart utilities — save helpers with kaleido timeout protection.
"""

from pathlib import Path
import signal
import os

CHARTS_DIR = Path(__file__).resolve().parent.parent / "charts_output"
CHARTS_DIR.mkdir(exist_ok=True)


def save_plotly(fig, name):
    """Save Plotly figure as HTML. PNG export skipped on Windows (kaleido issue)."""
    fig.write_html(str(CHARTS_DIR / f"{name}.html"))


def save_matplotlib(fig, name):
    """Save Matplotlib figure as PNG."""
    import matplotlib.pyplot as plt
    fig.savefig(str(CHARTS_DIR / f"{name}.png"), dpi=150, bbox_inches="tight",
                facecolor="#0e1117")
    plt.close(fig)


def axis_config(title, **extra):
    """Standard dark-theme axis config for Plotly charts."""
    base = dict(
        title=dict(text=title, font=dict(color="#e0e0e0")),
        gridcolor="#333",
        tickfont=dict(color="#aaa"),
    )
    base.update(extra)
    return base
