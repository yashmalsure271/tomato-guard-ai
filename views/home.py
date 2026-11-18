from __future__ import annotations

import streamlit as st

from utils.artifacts import dataset_stats, original_best, original_run
from utils.config import PAPER_DOI
from utils.constants import CLASS_NAMES
from utils.ui import (
    compact_html,
    escape,
    image_data_uri,
    pill,
    render_footer,
    section_head,
    stat_grid,
)

st.markdown(
    """
    <div class="tg-eyebrow">IA-2 machine learning project</div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    best = original_best()
    best_accuracy = f"{float(best['Accuracy']) * 100:.2f}%" if best else "99.70%"
    st.markdown(
        f"""
        <div class="tg-hero-copy">
          <h1>Spot leaf disease<br><em>before it spreads.</em></h1>
          <p class="tg-lead">
            Upload a close-up of a tomato leaf. TomatoGuard AI compares it with
            visual patterns learned from 11,000 labelled leaf images across 10
            classes, then returns an explainable screening result in seconds.
          </p>
          <div class="tg-badges">
            {pill("10 disease, pest and healthy classes")}
            {pill("ConvNeXt-Tiny")}
            {pill("Grad-CAM explanations")}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    action_left, action_right = st.columns(2)
    with action_left:
        if st.button(
            "Analyze a leaf",
            type="primary",
            icon=":material/biotech:",
            use_container_width=True,
        ):
            st.switch_page("views/prediction.py")
    with action_right:
        if st.button(
            "Explore the model",
            icon=":material/insights:",
            use_container_width=True,
        ):
            st.switch_page("views/dashboard.py")
    st.markdown(
        f"""
        <div class="tg-hero-note">
          Research-backed pipeline. Original notebook results preserved. Uploaded
          images are processed in memory and are never stored.
        </div>
        <div class="tg-stat" style="margin-top:1.4rem;max-width:270px;">
          <strong>{best_accuracy}</strong>
          <span>Best test accuracy in the original notebook run (ConvNeXt-Tiny)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    hero_image = image_data_uri("assets/samples/early_blight.png")
    st.markdown(
        f"""
        <div class="tg-hero-media">
          <img class="tg-hero-img" src="{hero_image}" alt="Tomato leaf with early blight symptoms">
          <div class="tg-float tg-float-top">
            <small>Dataset</small>
            <strong>11,000 leaf images</strong>
          </div>
          <div class="tg-float tg-float-bottom">
            <small>Deployment model</small>
            <strong><span class="tg-live"></span>ConvNeXt-Tiny</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

stats = dataset_stats()
st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
stat_grid(
    [
        {"value": f"{stats['total_images']:,}" if stats else "11,000", "label": "Labeled leaf images"},
        {"value": str(len(CLASS_NAMES)), "label": "Disease, pest and healthy classes"},
        {"value": "1,650", "label": "Held-out test images"},
        {"value": "4", "label": "CNN architectures compared"},
    ]
)

st.markdown('<div class="tg-section"></div>', unsafe_allow_html=True)
section_head(
    "The problem",
    "Expert diagnosis does not scale with the pace of crop loss.",
    "Tomato leaves show overlapping visual symptoms. A grower may not notice the first signs, and a delayed diagnosis can mean a lost crop.",
)
problem_left, problem_right = st.columns(2, gap="medium")
with problem_left:
    st.markdown(
        """
        <div class="tg-card">
          <div class="tg-kicker">Problem statement</div>
          <h3>Manual inspection is slow and subjective.</h3>
          <p>
            Tomato plants are affected by fungal, bacterial and viral diseases plus
            pests such as spider mites. Early symptoms are easy to confuse, and
            expert support is not always available when a grower needs it.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with problem_right:
    st.markdown(
        """
        <div class="tg-card">
          <div class="tg-kicker">Project objective</div>
          <h3>Make a fast, explainable second opinion accessible.</h3>
          <p>
            Build an image classifier that recognizes 10 tomato leaf classes and
            present the result through a deployable web application that also shows
            what evidence influenced the model.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="tg-section"></div>', unsafe_allow_html=True)
section_head(
    "How it works",
    "A screenshot-friendly flow from photo to screening result.",
)
step_one, step_two, step_three = st.columns(3, gap="medium")
with step_one:
    st.markdown(
        """
        <div class="tg-step">
          <div class="tg-step-num">01</div>
          <h3>Capture the leaf</h3>
          <p>
            Upload a photo or take one with a phone. One leaf, good light and a
            clean background give the best signal.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with step_two:
    st.markdown(
        """
        <div class="tg-step">
          <div class="tg-step-num">02</div>
          <h3>Classify the image</h3>
          <p>
            The image is resized to 224 x 224, normalized and passed through the
            trained ConvNeXt-Tiny classifier. The checkpoint loads once per server,
            not once per visit.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with step_three:
    st.markdown(
        """
        <div class="tg-step">
          <div class="tg-step-num">03</div>
          <h3>Review and explain</h3>
          <p>
            See the top three probabilities, a plain-language explanation, Grad-CAM
            attention and practical next steps. Download the result as a PDF.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="tg-section"></div>', unsafe_allow_html=True)
section_head(
    "Model comparison",
    "Four CNN architectures, one shared test split.",
    "These are the metrics and training times reported in the original notebook run. The deployed checkpoint is evaluated separately on the same seeded split.",
)
original = original_run()
if original and original.get("comparison"):
    rows = original["comparison"]
    best_f1 = max(float(row["F1"]) for row in rows)
    table_rows = []
    for row in rows:
        is_best = float(row["F1"]) == best_f1
        classes = ' class="tg-best"' if is_best else ""
        tag = ' <span class="tg-tag">Best F1</span>' if is_best else ""
        table_rows.append(
            compact_html(
                f"""
                <tr{classes}>
                  <td><strong>{escape(row['Model'])}</strong>{tag}</td>
                  <td class="tg-num">{float(row['Parameters (M)']):.2f}M</td>
                  <td class="tg-num">{float(row['Accuracy']) * 100:.2f}%</td>
                  <td class="tg-num">{float(row['Precision']) * 100:.2f}%</td>
                  <td class="tg-num">{float(row['Recall']) * 100:.2f}%</td>
                  <td class="tg-num">{float(row['F1']) * 100:.2f}%</td>
                  <td class="tg-num">{float(row['AUC']):.4f}</td>
                  <td class="tg-num">{float(row['Training Time (s)']):.0f}s</td>
                </tr>
                """
            )
        )
    st.markdown(
        f"""
        <div class="tg-table-wrap">
          <table class="tg-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Parameters</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>AUC</th>
                <th>Training time</th>
              </tr>
            </thead>
            <tbody>
              {''.join(table_rows)}
            </tbody>
          </table>
        </div>
        <div class="tg-source-note">
          Source: test-set evaluation saved in the original Colab notebook output,
          stratified 1,650-image test split. These values are preserved exactly.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("Original notebook comparison data is not available.")

st.markdown('<div class="tg-section"></div>', unsafe_allow_html=True)
paper_left, paper_right = st.columns([1.15, 0.85], gap="large")
with paper_left:
    st.markdown(
        """
        <div class="tg-quote">
          <div class="tg-kicker">Research foundation</div>
          <h3>A Lightweight Dual-Attention Network for Tomato Leaf Disease Identification</h3>
          <p>
            Enxu Zhang, Ning Zhang, Fei Li and Cheng Lv. Frontiers in Plant Science,
            2024. The paper's LDAMNet architecture is the lightweight baseline in this
            comparison study.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="tg-source-note"><a href="{PAPER_DOI}" target="_blank" rel="noopener">Open the paper (DOI: 10.3389/fpls.2024.1420584)</a></div>',
        unsafe_allow_html=True,
    )
with paper_right:
    st.markdown(
        """
        <div class="tg-card">
          <div class="tg-kicker">What was compared</div>
          <h3>Lightweight paper model vs. conventional and modern CNNs.</h3>
          <ul>
            <li>LDAMNet with channel and spatial attention</li>
            <li>ResNet50 as a conventional reference</li>
            <li>EfficientNetV2-S for efficient scaling</li>
            <li>ConvNeXt-Tiny as a modern CNN</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="tg-cta">
      <div class="tg-eyebrow">Ready when you are</div>
      <h2>Turn a leaf photo into an explainable screening result.</h2>
      <p>
        No sign-up, no image storage and no model retraining per visitor. The demo
        runs on CPU and is ready for a project presentation.
      </p>
      <a class="tg-cta-button" href="/prediction" target="_self">Get started</a>
    </div>
    """,
    unsafe_allow_html=True,
)
render_footer()
