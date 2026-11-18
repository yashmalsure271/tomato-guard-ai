from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.artifacts import (
    dataset_stats,
    fp16_equivalence,
    misclassifications,
    model_card,
    original_run,
    test_metrics,
    training_history,
)
from utils.config import DATASET_URL, ROOT
from utils.constants import CLASS_DISPLAY, CLASS_NAMES
from utils.ui import (
    CHART_COLORS,
    compact_html,
    CHART_LABELS,
    escape,
    metric_grid,
    page_header,
    plotly_theme,
    render_footer,
    sample_gallery,
    section_head,
    show_plot,
    stat_grid,
    tag,
)

page_header(
    "Model dashboard",
    "Dashboard",
    "Dataset facts, deployed-checkpoint metrics and the original notebook comparison, kept separate so every number has a clear source.",
)

metrics = test_metrics()
original = original_run()
stats = dataset_stats()
history = training_history()
card = model_card()
equivalence = fp16_equivalence()

data_tab, deployed_tab, original_tab = st.tabs(
    ["Data overview", "Deployed model", "Original notebook"]
)

with data_tab:
    if not stats:
        st.info("Dataset statistics are not available yet.")
    else:
        split = stats["splits"]
        image_size = next(iter(stats.get("image_sizes", {})), "256x256")
        stat_grid(
            [
                {"value": f"{stats['total_images']:,}", "label": "Labelled images"},
                {"value": str(stats["num_classes"]), "label": "Balanced classes"},
                {
                    "value": "70 / 15 / 15",
                    "label": "Train / validation / test split",
                },
                {"value": image_size, "label": "Native image resolution"},
            ]
        )
        st.markdown('<div class="tg-section"></div>', unsafe_allow_html=True)
        section_head(
            "Class distribution",
            "The dataset is balanced: 1,100 images for every class.",
            "A balanced target makes accuracy easier to interpret and prevents one class from dominating the loss.",
        )
        counts = stats["class_counts"]
        labels = [CHART_LABELS[name] for name in CLASS_NAMES]
        values = [counts[name] for name in CLASS_NAMES]
        dist_col, split_col = st.columns([1.35, 0.65], gap="large")
        with dist_col:
            figure = go.Figure(
                go.Bar(
                    x=values,
                    y=labels,
                    orientation="h",
                    marker={
                        "color": values,
                        "colorscale": [[0, "#c7e5c0"], [1, "#15803d"]],
                        "showscale": False,
                    },
                    text=values,
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="%{y}: %{x} images<extra></extra>",
                )
            )
            plotly_theme(figure, height=430, showlegend=False)
            figure.update_layout(
                margin={"l": 150, "r": 44, "t": 8, "b": 8},
                xaxis={"range": [0, 1300], "visible": False},
                yaxis={"showgrid": False, "autorange": "reversed"},
            )
            show_plot(figure, key="dataset_class_distribution")
        with split_col:
            split_values = [split["train"], split.get("val", split.get("validation")), split["test"]]
            figure = go.Figure(
                go.Pie(
                    labels=["Train", "Validation", "Test"],
                    values=split_values,
                    hole=0.64,
                    marker={"colors": ["#15803d", "#84cc16", "#d9f99d"]},
                    textinfo="percent",
                    textfont={"size": 11, "color": "#10241a"},
                    hovertemplate="%{label}: %{value:,} images<extra></extra>",
                )
            )
            plotly_theme(figure, height=430, showlegend=True)
            figure.update_layout(
                margin={"l": 8, "r": 8, "t": 20, "b": 20},
                annotations=[
                    {
                        "text": f"{stats['total_images']:,}<br><span style='font-size:11px;color:#60766a'>images</span>",
                        "x": 0.5,
                        "y": 0.5,
                        "showarrow": False,
                        "font": {"size": 20, "color": "#10241a"},
                    }
                ],
            )
            show_plot(figure, key="dataset_split")
        st.markdown(
            f"""
            <div class="tg-source-note">
              Split strategy: {escape(split["strategy"])} with seed {split["seed"]}
              ({split['train']:,} / {split['val'] if 'val' in split else split['validation']:,} / {split['test']:,} images).
              Native size: {escape(image_size.replace("x", " x "))}, mode:
              {escape(next(iter(stats.get("image_modes", {})), "RGB"))}.
              Source: <a href="{DATASET_URL}" target="_blank" rel="noopener">Hugging Face dataset card</a>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="tg-section"></div>', unsafe_allow_html=True)
        section_head(
            "Sample images",
            "One representative leaf from each class.",
            "Images are pulled from the same public dataset used for training and evaluation.",
        )
        st.markdown(sample_gallery(), unsafe_allow_html=True)

        st.markdown('<div class="tg-section"></div>', unsafe_allow_html=True)
        section_head("Pipeline", "Identical preprocessing for training and deployment.")
        pipe_left, pipe_right, pipe_third = st.columns(3, gap="medium")
        with pipe_left:
            st.markdown(
                """
                <div class="tg-card">
                  <div class="tg-kicker">Training only</div>
                  <h3>Augmentation</h3>
                  <ul>
                    <li>Resize to 224 x 224</li>
                    <li>Random horizontal flip</li>
                    <li>Random rotation up to 15 degrees</li>
                    <li>Color jitter: brightness, contrast, saturation 0.15</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with pipe_right:
            st.markdown(
                """
                <div class="tg-card">
                  <div class="tg-kicker">Training and inference</div>
                  <h3>Normalization</h3>
                  <ul>
                    <li>Resize to 224 x 224</li>
                    <li>Convert to tensor</li>
                    <li>ImageNet mean [0.485, 0.456, 0.406]</li>
                    <li>ImageNet standard deviation [0.229, 0.224, 0.225]</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with pipe_third:
            st.markdown(
                """
                <div class="tg-card">
                  <div class="tg-kicker">Target</div>
                  <h3>10-class label</h3>
                  <ul>
                    <li>One label per leaf image</li>
                    <li>9 disease or pest classes</li>
                    <li>1 healthy class</li>
                    <li>No multi-label or severity grading</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

with deployed_tab:
    if not metrics:
        st.warning(
            "Deployed checkpoint metrics are not available yet. Run the training script to generate the weights and evaluation artifacts."
        )
    else:
        values = metrics["metrics"]
        metric_grid(
            [
                {"label": "Accuracy", "value": f"{values['Accuracy'] * 100:.2f}%"},
                {"label": "Precision", "value": f"{values['Precision'] * 100:.2f}%"},
                {"label": "Recall", "value": f"{values['Recall'] * 100:.2f}%"},
                {"label": "F1 score", "value": f"{values['F1'] * 100:.2f}%"},
                {"label": "ROC-AUC (weighted OVR)", "value": f"{values['AUC']:.4f}"},
            ]
        )
        params = card.get("parameters") if card else None
        seconds = history.get("training_seconds") if history else None
        best_val = history.get("best_val_accuracy") if history else None
        info_cols = st.columns(4)
        info_cols[0].metric("Parameters", f"{params / 1_000_000:.2f}M" if params else "Pending")
        info_cols[1].metric(
            "Checkpoint size",
            f"{metrics.get('checkpoint_size_bytes', 0) / 1_000_000:.1f} MB"
            if metrics.get("checkpoint_size_bytes")
            else "Pending",
        )
        info_cols[2].metric(
            "Training time",
            f"{seconds / 60:.1f} min" if seconds else "Pending",
        )
        info_cols[3].metric(
            "Best validation accuracy",
            f"{best_val * 100:.2f}%" if best_val else "Pending",
        )
        st.markdown(
            f"""
            <div class="tg-source-note" style="margin-top:0.4rem;">
              Deployed checkpoint: {escape(str(metrics.get("model", "ConvNeXt-Tiny")))},
              evaluated on {metrics.get("test_size", 1650):,} held-out images with seed
              {metrics.get("split_seed", 42)}. Storage precision:
              {escape(str(metrics.get("storage_precision", "float32")))}
              {tag("Retrained", "")}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            "This checkpoint was retrained from the original notebook pipeline because the original .pth files were not preserved. "
            "Its metrics may differ slightly from the original Colab run. The original results are preserved separately in the next tab."
        )
        if equivalence:
            if equivalence.get("identical_predictions"):
                st.success(
                    f"Checkpoint storage check: the fp16 file produced identical predictions on all "
                    f"{equivalence.get('test_size', 1650):,} test images. Metrics above are for the exact deployed file."
                )
            else:
                st.warning(
                    f"Storage check: fp16 compression changed {equivalence.get('changed_predictions', 0)} test predictions. "
                    "Metrics above reflect the deployed checkpoint, not the original fp32 weights."
                )

        if history:
            epochs = list(range(1, len(history["history"]["train_acc"]) + 1))
            hist = history["history"]
            figure = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Accuracy", "Loss"),
                horizontal_spacing=0.09,
            )
            figure.add_trace(
                go.Scatter(
                    x=epochs,
                    y=hist["train_acc"],
                    mode="lines+markers",
                    name="Train accuracy",
                    line={"color": "#15803d", "width": 2.4},
                    marker={"size": 5},
                    hovertemplate="Epoch %{x}<br>Train accuracy: %{y:.2%}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=epochs,
                    y=hist["val_acc"],
                    mode="lines+markers",
                    name="Validation accuracy",
                    line={"color": "#84cc16", "width": 2.4},
                    marker={"size": 5},
                    hovertemplate="Epoch %{x}<br>Validation accuracy: %{y:.2%}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=epochs,
                    y=hist["train_loss"],
                    mode="lines+markers",
                    name="Train loss",
                    line={"color": "#b45309", "width": 2.4},
                    marker={"size": 5},
                    hovertemplate="Epoch %{x}<br>Train loss: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=2,
            )
            figure.add_trace(
                go.Scatter(
                    x=epochs,
                    y=hist["val_loss"],
                    mode="lines+markers",
                    name="Validation loss",
                    line={"color": "#d9482f", "width": 2.4},
                    marker={"size": 5},
                    hovertemplate="Epoch %{x}<br>Validation loss: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=2,
            )
            plotly_theme(figure, height=400)
            figure.update_layout(hovermode="x unified", margin={"l": 62, "r": 12, "t": 40, "b": 12})
            figure.update_yaxes(tickformat=".0%", row=1, col=1)
            figure.update_xaxes(title_text="Epoch", row=1, col=1)
            figure.update_xaxes(title_text="Epoch", row=1, col=2)
            figure.update_yaxes(title_text="Loss", row=1, col=2)
            show_plot(figure, key="deployed_training_curves")

        st.markdown('<div class="tg-section"></div>', unsafe_allow_html=True)
        section_head(
            "Confusion matrix",
            "Where the deployed model succeeds and where it confuses classes.",
            "Rows are actual classes and columns are predictions. Values are counts on the 1,650-image test split.",
        )
        cm = metrics["confusion_matrix"]
        matrix = np.asarray(cm["matrix"])
        text = [[str(int(value)) if value else "" for value in row] for row in matrix]
        short_labels = [CHART_LABELS[name] for name in cm["labels"]]
        figure = go.Figure(
            go.Heatmap(
                z=matrix,
                x=short_labels,
                y=short_labels,
                text=text,
                texttemplate="%{text}",
                textfont={"size": 9, "color": "#10241a"},
                colorscale=[
                    [0.0, "#f7fbf6"],
                    [0.35, "#dcefd8"],
                    [0.7, "#a7d29e"],
                    [1.0, "#6faa66"],
                ],
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Images: %{z}<extra></extra>",
                showscale=True,
                colorbar={"thickness": 10, "len": 0.7},
            )
        )
        plotly_theme(figure, height=650)
        figure.update_layout(
            margin={"l": 155, "r": 20, "t": 12, "b": 95},
            xaxis={"tickangle": -35, "title": "Predicted class", "showgrid": False},
            yaxis={"autorange": "reversed", "title": "Actual class", "showgrid": False},
        )
        show_plot(figure, key="deployed_confusion_matrix")

        top_confusions = metrics.get("top_confusions", [])
        if top_confusions:
            st.markdown(
                """
                <div class="tg-section-head" style="margin-top:1.4rem;">
                  <div>
                    <div class="tg-eyebrow">Error analysis</div>
                    <h2 style="font-size:1.3rem;">Most frequent confusions</h2>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            rows = "".join(
                f"<tr><td>{escape(CLASS_DISPLAY.get(item['actual'], item['actual']))}</td>"
                f"<td>{escape(CLASS_DISPLAY.get(item['predicted'], item['predicted']))}</td>"
                f"<td class='tg-num'>{item['count']}</td></tr>"
                for item in top_confusions
            )
            st.markdown(
                f"""
                <div class="tg-table-wrap">
                  <table class="tg-table">
                    <thead><tr><th>Actual</th><th>Predicted as</th><th>Images</th></tr></thead>
                    <tbody>{rows}</tbody>
                  </table>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.success("No off-diagonal confusion counts on this test split.")

        st.markdown('<div class="tg-section"></div>', unsafe_allow_html=True)
        section_head(
            "Per-class performance",
            "Precision, recall and F1 for each class.",
            "Support is 165 test images per class. Hover any bar for exact values.",
        )
        per_class = metrics["per_class"]
        labels_pc = [CHART_LABELS[name] for name in CLASS_NAMES]
        figure = go.Figure()
        for metric_name, color in [
            ("precision", "#15803d"),
            ("recall", "#84cc16"),
            ("f1-score", "#b45309"),
        ]:
            figure.add_trace(
                go.Bar(
                    name=metric_name.capitalize(),
                    x=labels_pc,
                    y=[per_class[name][metric_name] for name in CLASS_NAMES],
                    marker_color=color,
                    hovertemplate="%{x}<br>" + metric_name.capitalize() + ": %{y:.4f}<extra></extra>",
                )
            )
        plotly_theme(figure, height=430)
        figure.update_layout(
            barmode="group",
            bargap=0.25,
            bargroupgap=0.08,
            yaxis={"range": [0, 1.08], "tickformat": ".0%", "title": "Score"},
            xaxis={"tickangle": -30},
            margin={"l": 60, "r": 12, "t": 34, "b": 105},
        )
        show_plot(figure, key="deployed_per_class")

        st.markdown('<div class="tg-section"></div>', unsafe_allow_html=True)
        section_head(
            "ROC curves",
            "One-vs-rest ROC curves for all 10 classes.",
            "The weighted macro AUC is shown above. Curves are computed from the deployed checkpoint's test probabilities.",
        )
        figure = go.Figure()
        for index, name in enumerate(CLASS_NAMES):
            curve = metrics["roc_curves"].get(name)
            if not curve:
                continue
            figure.add_trace(
                go.Scatter(
                    x=curve["fpr"],
                    y=curve["tpr"],
                    mode="lines",
                    name=f"{CHART_LABELS[name]} (AUC {curve['auc']:.4f})",
                    line={"color": CHART_COLORS[index % len(CHART_COLORS)], "width": 2},
                    hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
                )
            )
        figure.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line={"dash": "dot", "color": "rgba(96,118,106,0.5)", "width": 1},
        )
        plotly_theme(figure, height=560)
        figure.update_layout(
            xaxis={"range": [0, 1], "title": "False positive rate", "tickformat": ".0%"},
            yaxis={"range": [0, 1.02], "title": "True positive rate", "tickformat": ".0%"},
            legend={
                "orientation": "v",
                "x": 1.02,
                "y": 1,
                "xanchor": "left",
                "yanchor": "top",
                "font": {"size": 9, "color": "#60766a"},
                "title": {"text": "Class (AUC)", "font": {"size": 9}},
            },
            margin={"l": 60, "r": 190, "t": 20, "b": 40},
        )
        show_plot(figure, key="deployed_roc")

        errors = misclassifications()
        if errors and errors.get("items"):
            with st.expander(f"View misclassified test images ({errors.get('count', 0)})"):
                st.caption(
                    "Test images where the deployed checkpoint disagreed with the dataset label. "
                    "Up to 12 examples are shown. Labels come from the public dataset and may "
                    "not always reflect field truth."
                )
                columns = st.columns(4)
                for index, item in enumerate(errors["items"]):
                    path = ROOT / item["file"]
                    if not path.exists():
                        continue
                    with columns[index % 4]:
                        st.image(str(path), use_container_width=True)
                        st.caption(
                            f"Actual: {CLASS_DISPLAY.get(item['actual'], item['actual'])}  \n"
                            f"Predicted: {CLASS_DISPLAY.get(item['predicted'], item['predicted'])} "
                            f"({item['confidence'] * 100:.1f}%)"
                        )

        st.markdown(
            """
            <div class="tg-disclaimer">
              Feature importance is not defined for this CNN in the tabular sense.
              Grad-CAM attention maps are the supported explanation method and are
              available on the Prediction page.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if history and history.get("config"):
            with st.expander("Training configuration"):
                st.json(history["config"])

with original_tab:
    if not original or not original.get("comparison"):
        st.info("Original notebook results are not available.")
    else:
        st.markdown(
            """
            <div class="tg-card tg-card-flat" style="margin-bottom:1rem;">
              <div class="tg-kicker">Preserved evidence</div>
              <h3>The original 10-epoch comparison run, unchanged.</h3>
              <p>
                These values, plots and confusion matrices were extracted from the
                saved Google Colab notebook outputs. They are shown exactly as the
                original run produced them, separately from the retrained deployment
                checkpoint.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        rows = original["comparison"]
        best_accuracy = max(float(row["Accuracy"]) for row in rows)
        table_rows = []
        for row in rows:
            is_best = float(row["Accuracy"]) == best_accuracy
            cls = ' class="tg-best"' if is_best else ""
            badge = ' <span class="tg-tag">Best accuracy</span>' if is_best else ""
            table_rows.append(
                compact_html(f"""
                <tr{cls}>
                  <td><strong>{escape(row['Model'])}</strong>{badge}</td>
                  <td class="tg-num">{float(row['Parameters (M)']):.2f}M</td>
                  <td class="tg-num">{float(row['Accuracy']) * 100:.2f}%</td>
                  <td class="tg-num">{float(row['Precision']) * 100:.2f}%</td>
                  <td class="tg-num">{float(row['Recall']) * 100:.2f}%</td>
                  <td class="tg-num">{float(row['F1']) * 100:.2f}%</td>
                  <td class="tg-num">{float(row['AUC']):.4f}</td>
                  <td class="tg-num">{float(row['Training Time (s)']):.0f}s</td>
                </tr>
                """)
            )
        st.markdown(
            f"""
            <div class="tg-table-wrap">
              <table class="tg-table">
                <thead>
                  <tr>
                    <th>Model</th><th>Parameters</th><th>Accuracy</th>
                    <th>Precision</th><th>Recall</th><th>F1</th>
                    <th>AUC</th><th>Training time</th>
                  </tr>
                </thead>
                <tbody>{''.join(table_rows)}</tbody>
              </table>
            </div>
            <div class="tg-source-note">
              Source: saved Colab notebook output, 1,650-image stratified test split.
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_names = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
        model_names = [row["Model"] for row in rows]
        figure = go.Figure()
        for metric_name, color in zip(
            metric_names, ["#15803d", "#84cc16", "#0f766e", "#b45309", "#d97706"], strict=True
        ):
            figure.add_trace(
                go.Bar(
                    name=metric_name,
                    x=model_names,
                    y=[float(row[metric_name]) for row in rows],
                    marker_color=color,
                    hovertemplate="%{x}<br>" + metric_name + ": %{y:.4f}<extra></extra>",
                )
            )
        plotly_theme(figure, height=390)
        figure.update_layout(
            barmode="group",
            bargap=0.24,
            bargroupgap=0.06,
            yaxis={"range": [0.94, 1.005], "tickformat": ".1%", "title": "Score"},
            margin={"l": 75, "r": 12, "t": 34, "b": 30},
        )
        show_plot(figure, key="original_metric_comparison")

        scatter_col, time_col = st.columns(2, gap="large")
        with scatter_col:
            figure = go.Figure(
                go.Scatter(
                    x=[float(row["Parameters (M)"]) for row in rows],
                    y=[float(row["Accuracy"]) for row in rows],
                    mode="markers+text",
                    text=model_names,
                    textposition="top center",
                    textfont={"size": 10},
                    marker={
                        "size": 15,
                        "color": CHART_COLORS[: len(rows)],
                        "line": {"width": 1, "color": "#ffffff"},
                    },
                    hovertemplate="%{text}<br>Parameters: %{x:.2f}M<br>Accuracy: %{y:.4f}<extra></extra>",
                    cliponaxis=False,
                )
            )
            plotly_theme(figure, height=350, showlegend=False)
            figure.update_layout(
                xaxis={"title": "Parameters (millions)", "range": [-2, 34]},
                yaxis={"title": "Test accuracy", "tickformat": ".1%", "range": [0.95, 1.002]},
                margin={"l": 65, "r": 20, "t": 30, "b": 30},
            )
            show_plot(figure, key="original_params_accuracy")
        with time_col:
            figure = go.Figure(
                go.Bar(
                    x=model_names,
                    y=[float(row["Training Time (s)"]) / 60 for row in rows],
                    marker_color=["#15803d", "#65a30d", "#0f766e", "#b45309"],
                    text=[f"{float(row['Training Time (s)']) / 60:.1f} min" for row in rows],
                    textposition="outside",
                    hovertemplate="%{x}<br>%{y:.1f} minutes<extra></extra>",
                )
            )
            plotly_theme(figure, height=350, showlegend=False)
            max_minutes = max(float(row["Training Time (s)"]) / 60 for row in rows)
            figure.update_layout(
                xaxis={"title": ""},
                yaxis={
                    "title": "Training time (minutes)",
                    "range": [0, max_minutes * 1.18],
                },
                margin={"l": 60, "r": 12, "t": 20, "b": 30},
            )
            show_plot(figure, key="original_training_time")

        st.markdown('<div class="tg-section"></div>', unsafe_allow_html=True)
        section_head(
            "Original training evidence",
            "Saved curves and confusion matrices from Colab.",
            "Choose a model to inspect its original 10-epoch accuracy/loss curves and test-set confusion matrix.",
        )
        selected = st.selectbox("Model", model_names, key="original_evidence_model")
        slug = selected.replace("-", "_")
        curves_path = ROOT / "assets" / "original_run" / f"training_curves_{slug}.png"
        matrix_path = ROOT / "assets" / "original_run" / f"confusion_matrix_{slug}.png"
        evidence_left, evidence_right = st.columns(2, gap="large")
        with evidence_left:
            if curves_path.exists():
                st.image(
                    str(curves_path),
                    caption=f"{selected}: original accuracy and loss curves",
                    use_container_width=True,
                )
        with evidence_right:
            if matrix_path.exists():
                st.image(
                    str(matrix_path),
                    caption=f"{selected}: original test-set confusion matrix",
                    use_container_width=True,
                )
        reports = original.get("classification_reports", {})
        if selected in reports:
            with st.expander("Original classification report"):
                st.code(reports[selected], language=None)

        st.markdown(
            """
            <div class="tg-disclaimer">
              These are static outputs from the original notebook. They are not
              recomputed by this website and were not edited. The deployed
              checkpoint has its own metrics in the previous tab.
            </div>
            """,
            unsafe_allow_html=True,
        )

render_footer()
