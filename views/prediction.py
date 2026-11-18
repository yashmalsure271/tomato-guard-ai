from __future__ import annotations

import hashlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.artifacts import deployed_accuracy, format_percent
from utils.config import MODEL_PATH
from utils.disease_info import (
    DISCLAIMER,
    LOW_CONFIDENCE,
    class_info,
    confidence_advice,
    confidence_label,
    display_name,
    prediction_explanation,
)
from utils.gradcam import generate_gradcam
from utils.inference import load_uploaded, predict
from utils.model import load_model
from utils.pdf_report import build_prediction_report, report_filename
from utils.ui import (
    CHART_LABELS,
    callout,
    confidence_bar,
    escape,
    page_header,
    plotly_theme,
    render_footer,
    result_card,
    show_plot,
)

page_header(
    "Leaf diagnosis",
    "Prediction",
    "Upload or capture a single tomato leaf. The model returns the three most likely classes, a Grad-CAM attention map and a downloadable report.",
)

accuracy = deployed_accuracy()
accuracy_text = f"{accuracy * 100:.2f}% test accuracy" if accuracy else "Evaluation pending"
checkpoint_ready = MODEL_PATH.exists()
if not checkpoint_ready:
    st.warning(
        "The trained checkpoint is not available in this environment yet. "
        "Upload and analysis controls are shown, but prediction is disabled until "
        "model/convnext_tiny_best.pth is present."
    )
st.markdown(
    f"""
    <div class="tg-model-strip">
      <span class="tg-pill"><span class="tg-dot"></span>ConvNeXt-Tiny</span>
      <span>{escape(accuracy_text)}</span>
      <span>1,650 held-out images</span>
      <span>|</span>
      <span>10 classes</span>
      <span>|</span>
      <span>CPU inference with cached weights</span>
    </div>
    """,
    unsafe_allow_html=True,
)


def _top3_figure(result: dict):
    items = list(reversed(result["top3"]))
    labels = [CHART_LABELS.get(item["class_name"], item["display_name"]) for item in items]
    values = [item["probability"] * 100 for item in items]
    colors = ["#c7ddc0", "#9ccf94", "#15803d"]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            text=[f"{value:.1f}%" for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )
    plotly_theme(figure, height=235, showlegend=False)
    figure.update_layout(
        margin={"l": 150, "r": 52, "t": 10, "b": 8},
        xaxis={"range": [0, max(110, max(values) * 1.18)], "visible": False},
        yaxis={"showgrid": False, "tickfont": {"size": 11, "color": "#294235"}},
        bargap=0.34,
    )
    return figure


def _all_classes_figure(probabilities: list[float]):
    from utils.constants import CLASS_NAMES

    pairs = sorted(
        zip(CLASS_NAMES, probabilities),
        key=lambda item: item[1],
    )
    labels = [CHART_LABELS[name] for name, _ in pairs]
    values = [value * 100 for _, value in pairs]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": "#7fb878"},
            text=[f"{value:.2f}%" for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )
    plotly_theme(figure, height=380, showlegend=False)
    figure.update_layout(
        margin={"l": 150, "r": 60, "t": 10, "b": 8},
        xaxis={"range": [0, 110], "visible": False},
        yaxis={"showgrid": False, "tickfont": {"size": 10}},
    )
    return figure


def _render_result_details(state: dict) -> None:
    result = state["result"]
    info = class_info(result["class_name"])
    st.markdown('<div class="tg-section"></div>', unsafe_allow_html=True)

    detail_left, detail_right = st.columns([1.05, 0.95], gap="large")
    with detail_left:
        st.markdown(
            callout(
                "What this means",
                prediction_explanation(
                    result["class_name"], result["confidence"], result["top3"]
                ),
            ),
            unsafe_allow_html=True,
        )
        confidence_note = confidence_advice(result["confidence"])
        if confidence_note:
            st.markdown(
                callout("Confidence check", confidence_note, kind="warn"),
                unsafe_allow_html=True,
            )
        if result["kind"] == "healthy":
            st.markdown(
                callout(
                    "Keep monitoring",
                    "A healthy prediction is not a guarantee. Continue scouting, especially after humid or rainy weather.",
                    kind="success",
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                callout(
                    "Suggested next steps",
                    "Use these as general first steps, then confirm with a local agricultural extension or plant health adviser before applying any treatment.",
                    bullets=info["actions"],
                ),
                unsafe_allow_html=True,
            )
        if result["class_name"] == "Late Blight":
            st.markdown(
                callout(
                    "Act quickly",
                    "Late blight can spread rapidly. Isolate affected plants, avoid overhead watering and contact local extension support promptly.",
                    kind="danger",
                ),
                unsafe_allow_html=True,
            )
    with detail_right:
        signs = "".join(f"<li>{escape(sign)}</li>" for sign in info["signs"])
        st.markdown(
            f"""
            <div class="tg-card">
              <div class="tg-kicker">Typical visual signs</div>
              <h3>{escape(display_name(result["class_name"]))}</h3>
              <p>{escape(info["summary"])}</p>
              <ul>{signs}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("See all 10 class probabilities"):
            show_plot(
                _all_classes_figure(result["probabilities"]),
                key=f"all_classes_{state['source_id'][:10]}",
            )

    if state.get("overlay") is not None:
        st.markdown('<div class="tg-section"></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="tg-section-head">
              <div>
                <div class="tg-eyebrow">Explainability</div>
                <h2>Grad-CAM attention</h2>
              </div>
              <p>
                Warmer regions contributed more strongly to the predicted class.
                This is an attention map, not a segmentation of diseased tissue.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        original_col, overlay_col = st.columns(2, gap="large")
        with original_col:
            st.image(
                state["image"],
                caption="Uploaded leaf",
                use_container_width=True,
            )
        with overlay_col:
            st.image(
                state["overlay"],
                caption="Grad-CAM overlay for the predicted class",
                use_container_width=True,
            )
    else:
        st.info(
            "Grad-CAM could not be generated for this image, but the prediction above is still valid. "
            "Try a smaller or clearer leaf image."
        )

    st.markdown(
        f'<div class="tg-disclaimer">{escape(DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )


def _reset_single() -> None:
    st.session_state["single_nonce"] = st.session_state.get("single_nonce", 0) + 1
    st.session_state.pop("single_result", None)
    st.rerun()


def _single_tab() -> None:
    nonce = st.session_state.get("single_nonce", 0)
    input_col, result_col = st.columns([1, 1], gap="large")

    with input_col:
        source = st.radio(
            "Image source",
            ["Upload image", "Use camera"],
            horizontal=True,
            label_visibility="collapsed",
        )
        uploaded = None
        if source == "Upload image":
            uploaded = st.file_uploader(
                "Leaf image",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
                key=f"single_file_{nonce}",
            )
        else:
            uploaded = st.camera_input(
                "Take a photo of one leaf",
                key=f"single_camera_{nonce}",
                label_visibility="collapsed",
            )

        image = None
        digest = None
        file_name = ""
        if uploaded is not None:
            file_name = getattr(uploaded, "name", "camera.jpg")
            try:
                image, digest = load_uploaded(uploaded)
            except Exception as exc:
                st.error(
                    "This file could not be read as an image. Use a JPG, PNG or WebP photo of a single leaf. "
                    f"Details: {exc}"
                )
            if image is not None:
                preview = image.copy()
                preview.thumbnail((900, 900))
                st.image(preview, caption="Preview", use_container_width=True)
                st.caption("One leaf, good light, plain background works best.")
        else:
            st.markdown(
                """
                <div class="tg-placeholder" style="min-height:12rem;margin-top:0.6rem;">
                  <div>
                    <h3>Start with a leaf photo</h3>
                    <p>Upload an image or use the camera. The photo is analyzed in memory and is never stored.</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        can_analyze = image is not None and checkpoint_ready
        if st.button(
            "Analyze leaf",
            type="primary",
            icon=":material/biotech:",
            use_container_width=True,
            disabled=not can_analyze,
        ):
            try:
                with st.spinner("Analyzing leaf image..."):
                    model = load_model()
                    result = predict(image, model)
                    overlay = None
                    try:
                        overlay, _ = generate_gradcam(model, image, result["class_index"])
                    except Exception:
                        overlay = None
                    try:
                        pdf_bytes = build_prediction_report(
                            image, result, overlay, accuracy
                        )
                        pdf_name = report_filename()
                    except Exception:
                        pdf_bytes = None
                        pdf_name = None
                st.session_state["single_result"] = {
                    "source_id": digest,
                    "source_name": file_name,
                    "image": image,
                    "result": result,
                    "overlay": overlay,
                    "pdf": pdf_bytes,
                    "pdf_name": pdf_name,
                }
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    state = st.session_state.get("single_result")
    current = state if state and state.get("source_id") == digest else None

    with result_col:
        if current is None:
            st.markdown(
                """
                <div class="tg-placeholder">
                  <div>
                    <h3>Your screening result will appear here</h3>
                    <p>
                      The model shows the top three classes, confidence and an
                      attention map. It will not show a result until you press
                      Analyze leaf.
                    </p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            result = current["result"]
            st.markdown(
                result_card(
                    result["class_name"], result["confidence"], result["kind"]
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="display:flex;justify-content:space-between;align-items:center;margin:0.9rem 0 0.15rem;">
                  <span class="tg-kicker" style="margin:0;">Top three probabilities</span>
                  <span class="tg-tag">{escape(confidence_label(result["confidence"]))}</span>
                </div>
                {confidence_bar(result["confidence"])}
                """,
                unsafe_allow_html=True,
            )
            show_plot(
                _top3_figure(result),
                key=f"single_top3_{current['source_id'][:10]}",
            )
            download_col, reset_col = st.columns([1.35, 1])
            with download_col:
                if current.get("pdf"):
                    st.download_button(
                        "PDF report",
                        data=current["pdf"],
                        file_name=current["pdf_name"],
                        mime="application/pdf",
                        icon=":material/description:",
                        use_container_width=True,
                    )
                else:
                    st.caption("PDF report is unavailable for this image.")
            with reset_col:
                if st.button(
                    "Start over",
                    icon=":material/refresh:",
                    use_container_width=True,
                ):
                    _reset_single()
            st.caption(
                f"File: {current['source_name']} | Model: ConvNeXt-Tiny | "
                f"Test accuracy: {format_percent(accuracy)}"
            )

    if current is not None:
        _render_result_details(current)


def _batch_id(files) -> str:
    parts = [f"{getattr(file, 'name', '')}:{getattr(file, 'size', 0)}" for file in files]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _csv_safe(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()
    for column in ("File", "Status"):
        safe[column] = safe[column].astype(str).map(
            lambda value: "'" + value
            if value.startswith(("=", "+", "-", "@"))
            else value
        )
    return safe


def _reset_batch() -> None:
    st.session_state["batch_nonce"] = st.session_state.get("batch_nonce", 0) + 1
    st.session_state.pop("batch_results", None)
    st.session_state.pop("batch_id", None)
    st.rerun()


def _batch_tab() -> None:
    nonce = st.session_state.get("batch_nonce", 0)
    st.markdown(
        """
        <div class="tg-card tg-card-flat" style="margin-bottom:1rem;">
          <div class="tg-kicker">Batch screening</div>
          <h3>Classify up to 25 leaf photos at once.</h3>
          <p>
            Each image gets a top-1 class and confidence score. Batch mode skips
            Grad-CAM and the PDF report; use the single-leaf tab for those.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    files = st.file_uploader(
        "Leaf images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"batch_files_{nonce}",
    )
    if not files:
        st.info("Add one or more leaf photos to start a batch.")
        return
    if len(files) > 25:
        st.error("Please keep a batch to 25 images or fewer.")
        return
    if sum(getattr(file, "size", 0) for file in files) > 120_000_000:
        st.error("This batch is too large. Keep the combined upload under 120 MB.")
        return
    if not checkpoint_ready:
        st.warning("Batch prediction is disabled until the trained checkpoint is available.")
        return

    batch_id = _batch_id(files)
    if st.button(
        "Analyze batch",
        type="primary",
        icon=":material/playlist_add_check:",
        use_container_width=False,
    ):
        try:
            with st.spinner("Loading model and analyzing images..."):
                model = load_model()
            progress = st.progress(0.0, text="Starting batch...")
            rows = []
            for index, file in enumerate(files):
                name = getattr(file, "name", f"image_{index + 1}.jpg")
                try:
                    image, _ = load_uploaded(file)
                    result = predict(image, model)
                    rows.append(
                        {
                            "File": name,
                            "Prediction": result["display_name"],
                            "Kind": result["kind"].capitalize(),
                            "Confidence (%)": round(result["confidence"] * 100, 2),
                            "Status": (
                                "Healthy"
                                if result["kind"] == "healthy"
                                and result["confidence"] >= LOW_CONFIDENCE
                                else "Needs attention"
                            ),
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "File": name,
                            "Prediction": "Could not read image",
                            "Kind": "-",
                            "Confidence (%)": None,
                            "Status": str(exc)[:120],
                        }
                    )
                progress.progress(
                    (index + 1) / len(files),
                    text=f"Analyzed {index + 1} of {len(files)} images",
                )
            progress.empty()
            st.session_state["batch_results"] = rows
            st.session_state["batch_id"] = batch_id
        except Exception as exc:
            st.error(f"Batch analysis failed: {exc}")
            return

    rows = st.session_state.get("batch_results")
    if not rows or st.session_state.get("batch_id") != batch_id:
        st.caption("Press Analyze batch to run the deployed model on these files.")
        return

    frame = pd.DataFrame(rows)
    valid = frame[frame["Confidence (%)"].notna()]
    flagged = int((frame["Status"] == "Needs attention").sum())
    healthy = int((frame["Status"] == "Healthy").sum())
    average = float(valid["Confidence (%)"].mean()) if not valid.empty else 0.0
    metric_cols = st.columns(4)
    metric_cols[0].metric("Images", len(frame))
    metric_cols[1].metric("Healthy", healthy)
    metric_cols[2].metric("Needs attention", flagged)
    metric_cols[3].metric("Average confidence", f"{average:.1f}%")
    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confidence (%)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    action_col, chart_col = st.columns([1, 1.3], gap="large")
    with action_col:
        st.download_button(
            "Download results CSV",
            data=_csv_safe(frame).to_csv(index=False).encode("utf-8"),
            file_name="tomato_batch_results.csv",
            mime="text/csv",
            icon=":material/download:",
            use_container_width=True,
        )
        if st.button(
            "Clear batch",
            icon=":material/refresh:",
            use_container_width=True,
        ):
            _reset_batch()
        st.markdown(
            '<div class="tg-disclaimer" style="margin-top:0.8rem;">'
            "Batch results use the same deployed checkpoint. Low-confidence healthy "
            "calls are flagged for review. Confidence is not a calibrated probability. "
            "Images are processed in memory and are never stored on the server."
            "</div>",
            unsafe_allow_html=True,
        )
    with chart_col:
        counts = (
            valid.groupby("Prediction")
            .size()
            .reset_index(name="Images")
            .sort_values("Images")
        )
        if not counts.empty:
            figure = go.Figure(
                go.Bar(
                    x=counts["Images"],
                    y=counts["Prediction"],
                    orientation="h",
                    marker={"color": "#15803d"},
                    text=counts["Images"],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="%{y}: %{x} images<extra></extra>",
                )
            )
            plotly_theme(figure, height=300, showlegend=False)
            figure.update_layout(
                margin={"l": 150, "r": 30, "t": 10, "b": 8},
                xaxis={
                    "visible": False,
                    "range": [0, max(int(counts["Images"].max()) * 1.25, 1.5)],
                },
                yaxis={"showgrid": False},
            )
            show_plot(figure, key=f"batch_counts_{batch_id[:8]}")


single_tab, batch_tab = st.tabs(["Single leaf", "Batch upload"])
with single_tab:
    _single_tab()
with batch_tab:
    _batch_tab()

render_footer()
