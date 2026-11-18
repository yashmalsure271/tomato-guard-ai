from __future__ import annotations

import base64
import html
from functools import lru_cache
from pathlib import Path

import streamlit as st

from utils.artifacts import deployed_accuracy, evaluated_date
from utils.config import (
    CSS_PATH,
    DATASET_URL,
    LEAF_MARK_PATH,
    PAPER_DOI,
    PROJECT_NAME,
    PROJECT_TAGLINE,
)
from utils.constants import CLASS_DISPLAY

CHART_COLORS = [
    "#15803d",
    "#84cc16",
    "#0f766e",
    "#65a30d",
    "#047857",
    "#ca8a04",
    "#b45309",
    "#0d9488",
    "#4d7c0f",
    "#d97706",
]

CHART_LABELS = {
    "Bacterial Spot": "Bacterial Spot",
    "Early Blight": "Early Blight",
    "Healthy": "Healthy",
    "Late Blight": "Late Blight",
    "Leaf Mold": "Leaf Mold",
    "Septoria Leaf Spot": "Septoria Leaf Spot",
    "Spider Mites Two-spotted Spider Mite": "Spider Mites",
    "Target Spot": "Target Spot",
    "Tomato Mosaic Virus": "Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus": "Yellow Leaf Curl Virus",
}

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
    "scrollZoom": False,
}


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def compact_html(markup: str) -> str:
    return "".join(line.strip() for line in markup.splitlines() if line.strip())


def inject_css() -> None:
    css = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@lru_cache(maxsize=24)
def image_data_uri(path: str) -> str:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    payload = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def page_header(title: str, eyebrow: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="tg-page-head">
          <div class="tg-eyebrow">{escape(eyebrow)}</div>
          <h1>{escape(title)}</h1>
          {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_head(eyebrow: str, title: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="tg-section-head">
          <div>
            <div class="tg-eyebrow">{escape(eyebrow)}</div>
            <h2>{escape(title)}</h2>
          </div>
          {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_grid(items: list[dict[str, str]]) -> None:
    cards = "".join(
        compact_html(
            f"""
            <div class="tg-stat">
              <strong>{escape(item['value'])}</strong>
              <span>{escape(item['label'])}</span>
            </div>
            """
        )
        for item in items
    )
    st.markdown(f'<div class="tg-stat-grid">{cards}</div>', unsafe_allow_html=True)


def metric_grid(items: list[dict[str, str]]) -> None:
    cards = "".join(
        compact_html(
            f"""
            <div class="tg-metric">
              <strong>{escape(item['value'])}</strong>
              <span>{escape(item['label'])}</span>
            </div>
            """
        )
        for item in items
    )
    st.markdown(f'<div class="tg-metric-grid">{cards}</div>', unsafe_allow_html=True)


def pill(text: str, dot: bool = True) -> str:
    dot_html = '<span class="tg-dot"></span>' if dot else ""
    return f'<span class="tg-pill">{dot_html}{escape(text)}</span>'


def tag(text: str, kind: str = "") -> str:
    extra = f" tg-tag-{kind}" if kind in {"warn", "danger"} else ""
    return f'<span class="tg-tag{extra}">{escape(text)}</span>'


def callout(
    title: str,
    body: str,
    kind: str = "",
    bullets: list[str] | None = None,
) -> str:
    extra = f" tg-callout-{kind}" if kind in {"warn", "danger", "success"} else ""
    bullet_html = ""
    if bullets:
        bullet_html = "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in bullets) + "</ul>"
    return compact_html(
        f"""
        <div class="tg-callout{extra}">
          <h3>{escape(title)}</h3>
          <p>{escape(body)}</p>
          {bullet_html}
        </div>
        """
    )


def result_card(class_name: str, confidence: float, kind: str) -> str:
    display = CLASS_DISPLAY.get(class_name, class_name)
    kind_class = {
        "healthy": "tg-result-healthy",
        "pest": "tg-result-pest",
        "disease": "tg-result-disease",
    }.get(kind, "tg-result-disease")
    percent = max(0.0, min(100.0, confidence * 100))
    return compact_html(
        f"""
        <div class="tg-result-card {kind_class}">
          <div>
            <div class="tg-result-label">Most likely class</div>
            <h2>{escape(display)}</h2>
            <p>Model confidence: {percent:.1f}%</p>
          </div>
          <div class="tg-conf-ring" style="--p:{percent:.1f}">
            <div>
              <strong>{percent:.1f}%</strong>
              <span>confidence</span>
            </div>
          </div>
        </div>
        """
    )


def confidence_bar(confidence: float) -> str:
    percent = max(0.0, min(100.0, confidence * 100))
    return (
        '<div class="tg-confidence-bar">'
        f'<span style="width:{percent:.1f}%"></span>'
        "</div>"
    )


def sample_gallery() -> str:
    figures = []
    for class_name, display in CLASS_DISPLAY.items():
        from utils.constants import CLASS_SLUGS

        slug = CLASS_SLUGS[class_name]
        path = Path(__file__).resolve().parents[1] / "assets" / "samples" / f"{slug}.png"
        if path.exists():
            figures.append(
                compact_html(
                    f"""
                    <figure>
                      <img src="{image_data_uri(str(path))}" alt="{escape(display)} sample leaf">
                      <figcaption>{escape(display)}</figcaption>
                    </figure>
                    """
                )
            )
    return f'<div class="tg-gallery">{"".join(figures)}</div>'


def plotly_theme(figure, height: int | None = None, showlegend: bool | None = None):
    figure.update_layout(
        template=None,
        font={"family": "Inter, sans-serif", "color": "#294235", "size": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 12, "r": 12, "t": 34, "b": 12},
        colorway=CHART_COLORS,
        hoverlabel={
            "bgcolor": "#ffffff",
            "bordercolor": "rgba(21,60,38,0.18)",
            "font": {"family": "Inter, sans-serif", "size": 12, "color": "#10241a"},
        },
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 11, "color": "#60766a"},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        xaxis={
            "gridcolor": "rgba(21,60,38,0.07)",
            "zeroline": False,
            "linecolor": "rgba(21,60,38,0.12)",
            "tickfont": {"size": 11, "color": "#60766a"},
            "title": {"font": {"size": 11, "color": "#60766a"}},
        },
        yaxis={
            "gridcolor": "rgba(21,60,38,0.07)",
            "zeroline": False,
            "linecolor": "rgba(21,60,38,0.12)",
            "tickfont": {"size": 11, "color": "#60766a"},
            "title": {"font": {"size": 11, "color": "#60766a"}},
        },
    )
    if height:
        figure.update_layout(height=height)
    if showlegend is not None:
        figure.update_layout(showlegend=showlegend)
    return figure


def show_plot(figure, key: str, height: int | None = None) -> None:
    if height:
        figure.update_layout(height=height)
    st.plotly_chart(
        figure,
        use_container_width=True,
        config=PLOTLY_CONFIG,
        key=key,
        theme=None,
    )


def render_footer() -> None:
    accuracy = deployed_accuracy()
    date = evaluated_date()
    accuracy_text = f"{accuracy * 100:.2f}% test accuracy" if accuracy else "metrics pending"
    date_text = f"evaluated {date}" if date else "awaiting evaluation"
    st.sidebar.markdown(
        f"""
        <div class="tg-side-note">
          <strong>Deployed model</strong><br>
          ConvNeXt-Tiny, {accuracy_text}<br>
          <span class="tg-mono">{escape(date_text)}</span>
          <div class="tg-side-facts">
            <span>10 classes</span>
            <span>224 x 224</span>
            <span>CPU inference</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="tg-footer">
          <div>
            <strong>{escape(PROJECT_NAME)}</strong> - {escape(PROJECT_TAGLINE)}<br>
            IA-2 project. Screening support only; not a laboratory diagnosis.
          </div>
          <div>
            <a href="{PAPER_DOI}" target="_blank" rel="noopener">Research paper</a>
            &nbsp;|&nbsp;
            <a href="{DATASET_URL}" target="_blank" rel="noopener">Dataset</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
