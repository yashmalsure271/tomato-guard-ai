from __future__ import annotations

from utils.constants import CLASS_DISPLAY

CLASS_INFO = {
    "Bacterial Spot": {
        "kind": "disease",
        "summary": "A bacterial disease caused by Xanthomonas species that can reduce leaf area and fruit quality.",
        "signs": [
            "Small water-soaked or dark spots on leaves",
            "Spots may join together and develop yellow halos",
            "Premature leaf drop in severe cases",
        ],
        "actions": [
            "Remove heavily affected leaves and plant debris",
            "Avoid overhead watering and splashing between plants",
            "Keep tools clean and rotate tomato crops",
            "Use locally recommended copper-based or other labeled products only with extension guidance",
        ],
    },
    "Early Blight": {
        "kind": "disease",
        "summary": "A fungal disease caused by Alternaria solani that commonly starts on older, lower leaves.",
        "signs": [
            "Brown spots with concentric rings, like a target",
            "Yellowing around the spots",
            "Lower leaves affected first, moving upward",
        ],
        "actions": [
            "Remove lower infected leaves and fallen debris",
            "Mulch to reduce soil splash and improve airflow",
            "Water at the base and avoid wetting foliage",
            "Rotate crops and follow local extension guidance for labeled fungicides",
        ],
    },
    "Healthy": {
        "kind": "healthy",
        "summary": "The model did not detect the visual signature of an unhealthy leaf class in this image.",
        "signs": [
            "Relatively uniform green color",
            "No obvious lesions, mold or stippling detected by the model",
        ],
        "actions": [
            "Continue regular scouting of upper and lower leaf surfaces",
            "Keep watering and nutrition balanced for the crop stage",
            "Re-check after rain, high humidity or sudden temperature changes",
        ],
    },
    "Late Blight": {
        "kind": "disease",
        "summary": "A fast-spreading disease caused by Phytophthora infestans that can rapidly damage tomato foliage and fruit.",
        "signs": [
            "Large, irregular, greasy-looking green to brown lesions",
            "White fuzzy growth on the underside in humid conditions",
            "Rapid collapse of leaves and stems",
        ],
        "actions": [
            "Contact a local agricultural extension or plant health adviser quickly",
            "Remove and carefully dispose of infected plants",
            "Avoid overhead irrigation and keep foliage dry",
            "Protect nearby plants according to local guidance; this disease spreads fast",
        ],
    },
    "Leaf Mold": {
        "kind": "disease",
        "summary": "A fungal disease associated with high humidity, especially in greenhouses and tunnels.",
        "signs": [
            "Pale yellow or green spots on the upper leaf surface",
            "Olive-green to brown velvety mold on the underside",
            "Leaves may curl and drop in severe cases",
        ],
        "actions": [
            "Increase ventilation and reduce humidity around the canopy",
            "Avoid leaf wetness and remove affected debris",
            "Space plants and prune for better airflow",
            "Consider resistant varieties and follow local guidance for labeled treatments",
        ],
    },
    "Septoria Leaf Spot": {
        "kind": "disease",
        "summary": "A fungal disease caused by Septoria lycopersici that is common in humid conditions.",
        "signs": [
            "Small circular spots with gray centers and dark margins",
            "Spots often start on lower leaves",
            "Yellowing and leaf drop as the disease progresses",
        ],
        "actions": [
            "Remove infected leaves and plant debris",
            "Mulch and avoid overhead watering",
            "Rotate crops and improve airflow",
            "Follow local extension guidance for labeled fungicides if pressure is high",
        ],
    },
    "Spider Mites Two-spotted Spider Mite": {
        "kind": "pest",
        "summary": "A tiny pest that feeds on leaf undersides and can cause severe yellowing in hot, dry conditions.",
        "signs": [
            "Fine pale stippling or speckling on leaves",
            "Yellowing and drying of leaves",
            "Fine webbing on leaf undersides or growing tips",
        ],
        "actions": [
            "Inspect leaf undersides with a hand lens",
            "Reduce plant stress and keep plants well watered",
            "Use a strong water spray or labeled horticultural soap or oil",
            "Encourage beneficial insects and avoid unnecessary broad-spectrum sprays",
        ],
    },
    "Target Spot": {
        "kind": "disease",
        "summary": "A fungal disease caused by Corynespora cassiicola that is favored by warm, humid weather.",
        "signs": [
            "Brown lesions with concentric rings",
            "Spots may appear on leaves, stems and fruit",
            "Yellowing and leaf drop in severe cases",
        ],
        "actions": [
            "Remove affected leaves and improve airflow",
            "Avoid overhead irrigation and reduce leaf wetness",
            "Rotate crops and clean up plant debris",
            "Follow local extension guidance for labeled treatments",
        ],
    },
    "Tomato Mosaic Virus": {
        "kind": "disease",
        "summary": "A viral disease that can cause mottling and distortion and is easily spread by handling and tools.",
        "signs": [
            "Mosaic patterns of light and dark green",
            "Leaf distortion or fern-like growth",
            "Stunting and reduced fruit yield",
        ],
        "actions": [
            "Remove infected plants; viruses cannot be cured",
            "Disinfect pruning tools and wash hands between plants",
            "Use clean seed or transplants and resistant varieties",
            "Avoid handling healthy plants after infected ones",
        ],
    },
    "Tomato Yellow Leaf Curl Virus": {
        "kind": "disease",
        "summary": "A virus spread by whiteflies that causes leaf curling, yellowing and stunting.",
        "signs": [
            "Upward curling of leaf edges",
            "Yellowing between veins and stunting",
            "Reduced flowering and fruit set",
        ],
        "actions": [
            "Monitor and manage whiteflies with exclusion netting and sticky traps",
            "Remove infected plants to reduce virus sources",
            "Use resistant varieties where available",
            "Follow local guidance for labeled whitefly management",
        ],
    },
}

DISCLAIMER = (
    "This is an image-based screening tool built for a college project. It is not a "
    "laboratory diagnosis, and it does not replace advice from an agronomist or plant "
    "health professional. Always confirm findings in the field before applying treatments."
)

LOW_CONFIDENCE = 0.60
HIGH_CONFIDENCE = 0.85


def class_info(class_name: str) -> dict:
    fallback = {
        "kind": "disease",
        "summary": "No description is available for this class.",
        "signs": [],
        "actions": [],
    }
    return CLASS_INFO.get(class_name, fallback)


def display_name(class_name: str) -> str:
    return CLASS_DISPLAY.get(class_name, class_name)


def prediction_explanation(class_name: str, confidence: float, top3: list[dict]) -> str:
    info = class_info(class_name)
    name = display_name(class_name)
    parts = [
        f"The leaf's color, texture and lesion patterns look most similar to {name} "
        f"in the training images. The model assigned {confidence * 100:.1f}% of its "
        f"probability to this class."
    ]
    if len(top3) > 1:
        runner_up = top3[1]
        runner_name = display_name(runner_up["class_name"])
        runner_probability = runner_up["probability"]
        if runner_probability >= 0.15:
            parts.append(
                f"The next closest match is {runner_name} at {runner_probability * 100:.1f}%, "
                "so the image has some visual overlap between these classes."
            )
    parts.append(f"In simple terms: {info['summary']}")
    return " ".join(parts)


def confidence_label(confidence: float) -> str:
    if confidence >= HIGH_CONFIDENCE:
        return "High confidence"
    if confidence >= LOW_CONFIDENCE:
        return "Moderate confidence"
    return "Low confidence"


def confidence_advice(confidence: float) -> str | None:
    if confidence >= HIGH_CONFIDENCE:
        return None
    if confidence >= LOW_CONFIDENCE:
        return (
            "The model is not strongly certain. Take another photo in good light, "
            "with a single leaf filling most of the frame, and compare the result."
        )
    return (
        "Confidence is low. Avoid acting on this result alone. Use a clear, well-lit "
        "close-up of one leaf, avoid shadows and background clutter, and consult a "
        "local plant health expert if symptoms persist."
    )
