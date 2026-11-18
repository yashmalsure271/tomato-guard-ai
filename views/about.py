from __future__ import annotations

import streamlit as st

from utils.artifacts import model_card, original_run, test_metrics
from utils.config import (
    COURSE_NAME,
    DATASET_URL,
    PAPER_DOI,
    SOURCE_NOTEBOOK,
)
from utils.constants import CLASS_NAMES
from utils.ui import (
    compact_html,
    escape,
    page_header,
    render_footer,
    section_head,
    tag,
)

page_header(
    "About the project",
    "About",
    "A course project turned into a deployable screening tool: the objective, methodology, model choices, limitations and responsible-use notes.",
)

deck = original_run()
metrics = test_metrics()
card = model_card()

intro_left, intro_right = st.columns([1.25, 0.75], gap="large")
with intro_left:
    st.markdown(
        """
        <div class="tg-card">
          <div class="tg-kicker">Project context</div>
          <h3>Comparative CNN study for tomato leaf disease classification.</h3>
          <p>
            Tomato plants are affected by fungal, bacterial and viral diseases plus
            pests that can reduce yield and quality. Manual identification requires
            expert knowledge and is not always available at the moment a grower needs
            it. This project studies whether lightweight and modern CNN architectures
            can provide a fast, explainable screening signal from a single leaf photo.
          </p>
          <p style="margin-top:0.8rem;">
            The project began as an IA-2 comparative study of four CNN architectures.
            TomatoGuard AI turns the best-performing training run into a public web
            application with a preserved record of the original notebook results.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with intro_right:
    st.markdown(
        f"""
        <div class="tg-card">
          <div class="tg-kicker">At a glance</div>
          <h3>Project facts</h3>
          <ul>
            <li>Course: {escape(COURSE_NAME)}</li>
            <li>Application: plant disease classification</li>
            <li>Task: 10-class single-label image classification</li>
            <li>Dataset: {len(CLASS_NAMES) * 1100:,} images, 10 balanced classes</li>
            <li>Deployment: Streamlit on CPU</li>
            <li>Explainability: Grad-CAM attention maps</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="tg-section"></div>', unsafe_allow_html=True)
section_head(
    "Problem and objective",
    "From a leaf photo to a transparent screening result.",
)
problem_left, problem_right = st.columns(2, gap="medium")
with problem_left:
    st.markdown(
        """
        <div class="tg-card">
          <div class="tg-kicker">Problem statement</div>
          <h3>Symptoms overlap and expert support does not scale.</h3>
          <p>
            Early symptoms of bacterial spot, early blight, late blight, leaf mold
            and other classes can look similar. Waiting for a confident diagnosis can
            allow disease to spread, while unnecessary treatments waste cost and
            resources.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with problem_right:
    st.markdown(
        """
        <div class="tg-card">
          <div class="tg-kicker">Objective</div>
          <h3>Compare CNN architectures, then deploy the best one responsibly.</h3>
          <ul>
            <li>Reproduce the paper-inspired LDAMNet baseline</li>
            <li>Compare it with ResNet50, EfficientNetV2-S and ConvNeXt-Tiny</li>
            <li>Use one stratified split and consistent evaluation</li>
            <li>Deploy the best checkpoint with confidence and Grad-CAM explanations</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="tg-section"></div>', unsafe_allow_html=True)
section_head(
    "Methodology",
    "A reproducible pipeline, documented end to end.",
    "The original notebook remains the source of truth for the research run. The deployment reproduces its preprocessing and training logic.",
)
methods = st.columns(3, gap="medium")
with methods[0]:
    st.markdown(
        """
        <div class="tg-step">
          <div class="tg-step-num">01</div>
          <h3>Dataset and split</h3>
          <p>
            The Hugging Face dataset has 11,000 RGB images at 256 x 256, with 1,100
            images per class. The single train split was divided with a stratified
            70/15/15 split using seed 42: 7,700 train, 1,650 validation and 1,650
            test images.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with methods[1]:
    st.markdown(
        """
        <div class="tg-step">
          <div class="tg-step-num">02</div>
          <h3>Training</h3>
          <p>
            Images were resized to 224 x 224 and normalized with ImageNet statistics.
            Training used horizontal flips, 15 degree rotations and color jitter.
            Each model used Adam at 1e-4, batch size 32, 10 epochs, cross-entropy
            loss and the best validation checkpoint.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with methods[2]:
    st.markdown(
        """
        <div class="tg-step">
          <div class="tg-step-num">03</div>
          <h3>Evaluation and deployment</h3>
          <p>
            Accuracy, weighted precision, recall, F1 and one-vs-rest AUC were reported
            on the held-out test split. The deployed model uses the same resize and
            normalization steps, loads the checkpoint once per server and runs on CPU.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="tg-section"></div>', unsafe_allow_html=True)
section_head(
    "Models",
    "Four architectures, identical data and evaluation.",
    "Parameter counts and metrics below are from the original notebook run.",
)
if deck and deck.get("comparison"):
    rows = deck["comparison"]
    table_rows = []
    for row in rows:
        table_rows.append(
            compact_html(f"""
            <tr>
              <td><strong>{escape(row["Model"])}</strong></td>
              <td class="tg-num">{float(row["Parameters (M)"]):.2f}M</td>
              <td class="tg-num">{float(row["Accuracy"]) * 100:.2f}%</td>
              <td class="tg-num">{float(row["F1"]) * 100:.2f}%</td>
              <td class="tg-num">{float(row["AUC"]):.4f}</td>
              <td class="tg-num">{float(row["Training Time (s)"]):.0f}s</td>
            </tr>
            """)
        )
    st.markdown(
        f"""
        <div class="tg-table-wrap">
          <table class="tg-table">
            <thead>
              <tr><th>Model</th><th>Parameters</th><th>Accuracy</th><th>F1</th><th>AUC</th><th>Training time</th></tr>
            </thead>
            <tbody>{''.join(table_rows)}</tbody>
          </table>
        </div>
        <div class="tg-source-note">
          Original notebook test results on the shared 1,650-image test split.
          LDAMNet is a paper-inspired implementation of channel and spatial attention
          blocks, not a byte-for-byte reproduction of the published model.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("Original model comparison data is not available.")

deployed_left, deployed_right = st.columns([1, 1], gap="large")
with deployed_left:
    accuracy = (
        f"{metrics['metrics']['Accuracy'] * 100:.2f}%" if metrics else "Pending"
    )
    params = f"{card['parameters'] / 1_000_000:.2f}M" if card else "27.83M"
    st.markdown(
        f"""
        <div class="tg-card">
          <div class="tg-kicker">Live model</div>
          <h3>ConvNeXt-Tiny deployment checkpoint</h3>
          <ul>
            <li>Architecture: torchvision ConvNeXt-Tiny with a 10-class head</li>
            <li>Parameters: {escape(params)}</li>
            <li>Deployed test accuracy: {escape(accuracy)}</li>
            <li>Input: 224 x 224 RGB, ImageNet normalization</li>
            <li>Checkpoint: float16 storage loaded as float32 weights</li>
            <li>Inference: CPU, cached once per server process</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with deployed_right:
    st.markdown(
        f"""
        <div class="tg-callout tg-callout-warn">
          <h3>Retraining provenance {tag("Important", "warn")}</h3>
          <p>
            The original .pth files were not preserved with the notebook, so the
            live checkpoint was retrained from the same notebook pipeline on Apple
            MPS hardware.
          </p>
          <p style="margin-top:0.65rem;">
            This is why the deployed model has its own evaluation and may differ
            slightly from the original 99.70% ConvNeXt-Tiny result. The original
            notebook metrics, curves and confusion matrices remain preserved and
            clearly labeled in the Dashboard.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="tg-section"></div>', unsafe_allow_html=True)
section_head("Technology stack", "Built with a small, production-friendly Python stack.")
stack = [
    ("Python 3.12", "Application runtime"),
    ("PyTorch", "Model training and inference"),
    ("torchvision", "ConvNeXt-Tiny architecture and transforms"),
    ("Hugging Face datasets", "Dataset loading and caching"),
    ("scikit-learn", "Split, metrics and ROC"),
    ("Streamlit", "Web application and deployment"),
    ("Plotly", "Interactive charts"),
    ("Pandas / NumPy", "Tabular processing and numerics"),
    ("Pillow", "Image loading and preprocessing"),
    ("fpdf2", "Downloadable PDF reports"),
    ("GitHub", "Source control"),
    ("Streamlit Community Cloud", "Public hosting"),
]
stack_html = "".join(
    f'<span class="tg-pill"><span class="tg-dot"></span>{escape(name)}</span>'
    for name, _ in stack
)
st.markdown(f'<div class="tg-badges">{stack_html}</div>', unsafe_allow_html=True)

st.markdown('<div class="tg-section"></div>', unsafe_allow_html=True)
section_head(
    "Limitations and responsible use",
    "What this system can and cannot tell you.",
)
limits_left, limits_right = st.columns(2, gap="large")
with limits_left:
    st.markdown(
        """
        <div class="tg-card">
          <div class="tg-kicker">Model limits</div>
          <h3>Not a field diagnostic system.</h3>
          <ul>
            <li>Only 10 tomato leaf classes; no other crops, stems or fruit.</li>
            <li>No severity score and no multi-label diagnosis.</li>
            <li>Performance is measured on one public dataset, not on external farms.</li>
            <li>Image quality, lighting, background and camera differences can change predictions.</li>
            <li>Grad-CAM shows attention, not proof that a region is diseased.</li>
            <li>Softmax confidence is not a calibrated probability or a precise risk estimate.</li>
            <li>The deployed checkpoint is a retrain, not the original saved weights.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with limits_right:
    st.markdown(
        """
        <div class="tg-callout tg-callout-danger">
          <h3>Responsible use</h3>
          <p>
            Use this tool to support learning and early scouting, not as the sole
            basis for treatment. Confirm uncertain or high-impact findings with a
            local agricultural extension service, agronomist or plant clinic.
          </p>
          <p style="margin-top:0.65rem;">
            Always follow local regulations and product labels. Do not apply
            pesticides based only on a model prediction.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

privacy_left, privacy_right = st.columns(2, gap="large")
with privacy_left:
    st.markdown(
        """
        <div class="tg-card">
          <div class="tg-kicker">Privacy</div>
          <h3>Images stay in memory.</h3>
          <p>
            Uploaded and camera images are processed for the current request only.
            The app does not create accounts, write images to disk, or persist
            uploaded files. PDF reports are generated in memory and downloaded
            directly to your device. Streamlit usage statistics are disabled in the
            app configuration.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with privacy_right:
    st.markdown(
        """
        <div class="tg-card">
          <div class="tg-kicker">Known failure modes</div>
          <h3>Where to be extra careful.</h3>
          <ul>
            <li>Multiple leaves or busy backgrounds</li>
            <li>Blur, underexposure and strong color casts</li>
            <li>Very early symptoms that are not yet visible</li>
            <li>Look-alike nutrient deficiencies and chemical injury</li>
            <li>Classes underrepresented in real field conditions</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="tg-section"></div>', unsafe_allow_html=True)
section_head("Research and credits", "Sources that anchor the project.")
credit_left, credit_right = st.columns([1.2, 0.8], gap="large")
with credit_left:
    st.markdown(
        f"""
        <div class="tg-quote">
          <div class="tg-kicker">Paper</div>
          <h3>A Lightweight Dual-Attention Network for Tomato Leaf Disease Identification</h3>
          <p>
            Enxu Zhang, Ning Zhang, Fei Li and Cheng Lv. Frontiers in Plant Science,
            2024. DOI: 10.3389/fpls.2024.1420584. LDAMNet is the lightweight
            paper-inspired baseline in this comparison study.
          </p>
        </div>
        <div class="tg-source-note">
          <a href="{PAPER_DOI}" target="_blank" rel="noopener">Open the paper</a>
          &nbsp;|&nbsp;
          <a href="{DATASET_URL}" target="_blank" rel="noopener">Dataset card</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
with credit_right:
    st.markdown(
        f"""
        <div class="tg-card">
          <div class="tg-kicker">Preserved source</div>
          <h3>Original notebook included.</h3>
          <p>
            The original Colab notebook is stored in the repository at
            <span class="tg-mono">{escape(SOURCE_NOTEBOOK)}</span>. Its saved outputs
            are the source of the original comparison results and evidence images.
          </p>
          <p style="margin-top:0.65rem;">
            Dataset credit: Project-AgML/tomato_leaf_disease on Hugging Face. Check
            the dataset card for license and usage terms.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="tg-disclaimer">
      Educational project disclaimer: this website was built for an IA-2 machine
      learning course project and a public demonstration. Model outputs are
      probabilistic and can be wrong. No warranty is provided for agricultural,
      commercial or safety-critical use. Always confirm plant health decisions with
      a qualified local professional.
    </div>
    """,
    unsafe_allow_html=True,
)

render_footer()
