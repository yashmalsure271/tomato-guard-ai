from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

from fpdf import FPDF
from PIL import Image

from utils.config import LEAF_MARK_PATH, PROJECT_NAME
from utils.disease_info import (
    DISCLAIMER,
    class_info,
    confidence_advice,
    prediction_explanation,
    display_name,
)


def _latin(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")


def _square(image: Image.Image, size: int = 720) -> BytesIO:
    image = image.convert("RGB")
    side = min(image.width, image.height)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((size, size), Image.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)
    return buffer


class ReportPDF(FPDF):
    def header(self) -> None:
        if LEAF_MARK_PATH.exists():
            self.image(str(LEAF_MARK_PATH), x=15, y=10, w=11)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(21, 128, 61)
        self.set_xy(29, 12)
        self.cell(0, 5, _latin(PROJECT_NAME), new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(96, 118, 106)
        self.set_x(29)
        self.cell(0, 4, "Leaf image analysis report", new_x="LMARGIN", new_y="NEXT")

    def footer(self) -> None:
        self.set_y(-13)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(120, 137, 127)
        self.cell(
            0,
            5,
            "Screening support only - not a laboratory diagnosis. Page "
            + str(self.page_no()),
            align="C",
        )


def build_prediction_report(
    image: Image.Image,
    result: dict,
    overlay: Image.Image | None,
    model_accuracy: float | None,
) -> bytes:
    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(15, 15, 15)
    pdf.set_title("TomatoGuard AI leaf analysis report")
    pdf.set_author("TomatoGuard AI")
    pdf.set_subject("Tomato leaf image screening result")
    pdf.add_page()

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(96, 118, 106)
    pdf.cell(0, 5, _latin(f"Generated: {generated}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    class_name = result["class_name"]
    confidence = float(result["confidence"])
    kind = result["kind"]
    info = class_info(class_name)
    top3 = result.get("top3", [])

    if kind == "healthy":
        pdf.set_fill_color(240, 253, 244)
        pdf.set_text_color(6, 78, 59)
    else:
        pdf.set_fill_color(255, 247, 237)
        pdf.set_text_color(124, 45, 18)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(
        0,
        12,
        _latin(f"Most likely: {display_name(class_name)}"),
        fill=True,
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0,
        7,
        _latin(f"Model confidence: {confidence * 100:.1f}%"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    if model_accuracy is not None:
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(96, 118, 106)
        pdf.cell(
            0,
            6,
            _latin(
                f"Deployed ConvNeXt-Tiny test accuracy: {model_accuracy * 100:.2f}% "
                "(1,650 held-out images)"
            ),
            new_x="LMARGIN",
            new_y="NEXT",
        )
    pdf.ln(3)

    image_width = 82
    image_gap = 14
    image_y = pdf.get_y()
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(41, 66, 53)
    pdf.set_xy(15, image_y)
    pdf.cell(image_width, 5, "Uploaded leaf", align="C")
    if overlay is not None:
        pdf.set_xy(15 + image_width + image_gap, image_y)
        pdf.cell(image_width, 5, "Grad-CAM attention", align="C")
    original_buffer = _square(image)
    pdf.image(original_buffer, x=15, y=image_y + 5, w=image_width)
    if overlay is not None:
        overlay_buffer = _square(overlay)
        pdf.image(
            overlay_buffer,
            x=15 + image_width + image_gap,
            y=image_y + 5,
            w=image_width,
        )
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(96, 118, 106)
        pdf.set_xy(15 + image_width + image_gap, image_y + image_width + 7)
        pdf.cell(
            image_width,
            4,
            "Regions influencing the prediction",
            align="C",
        )
    pdf.set_y(image_y + image_width + 12)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(16, 36, 26)
    pdf.cell(0, 7, "Top three probabilities", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(246, 250, 245)
    pdf.cell(10, 7, "Rank", border=1, fill=True)
    pdf.cell(112, 7, "Class", border=1, fill=True)
    pdf.cell(0, 7, "Probability", border=1, align="R", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8.5)
    for rank, item in enumerate(top3, start=1):
        pdf.cell(10, 7, str(rank), border=1)
        pdf.cell(112, 7, _latin(display_name(item["class_name"])), border=1)
        pdf.cell(
            0,
            7,
            f"{item['probability'] * 100:.1f}%",
            border=1,
            align="R",
            new_x="LMARGIN",
            new_y="NEXT",
        )
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(16, 36, 26)
    pdf.cell(0, 7, "What this means", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(41, 66, 53)
    pdf.multi_cell(0, 5, _latin(info["summary"]))
    pdf.ln(1)
    pdf.multi_cell(
        0,
        5,
        _latin(prediction_explanation(class_name, confidence, top3)),
    )
    advice = confidence_advice(confidence)
    if advice:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(180, 83, 9)
        pdf.multi_cell(0, 5, _latin("Confidence note: " + advice))
        pdf.set_text_color(41, 66, 53)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(16, 36, 26)
    pdf.cell(0, 7, "Suggested next steps", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(41, 66, 53)
    for action in info["actions"]:
        pdf.multi_cell(0, 5, _latin("- " + action))
        pdf.ln(0.5)
    pdf.ln(2)

    if info.get("signs"):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(16, 36, 26)
        pdf.cell(0, 7, "Typical visual signs", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(41, 66, 53)
        for sign in info["signs"]:
            pdf.multi_cell(0, 5, _latin("- " + sign))
            pdf.ln(0.5)
        pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(16, 36, 26)
    pdf.cell(0, 7, "Important disclaimer", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(96, 118, 106)
    pdf.multi_cell(0, 4.5, _latin(DISCLAIMER))

    output = pdf.output()
    return bytes(output)


def report_filename() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"tomato_leaf_report_{stamp}.pdf"
