from transformers import pipeline
import pandas as pd

#open the document

df = pd.read_csv("test_titles.csv",
                 encoding="utf-8")

# Frame declaration
"""
FRAMES = [
    "Economic consequences",
    "Political strategy and power",
    "Security and defense",
    "Morality and ethics",
    "Fairness and equality",
    "Legal consequences",
    "Government policies",
    "Health and safety",
    "Public opinion",
    "Cultural identity"
]
"""

# Test df

# Clasificador

classifier = pipeline(
    task="zero-shot-classification",
    model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    device=-1
)

# Framing detection
"""
def detect_frames(title, threshold=0.5):

    if not isinstance(title, str) or not title.strip():
        return {
            "frames": [],
            "scores": {}
        }

    result = classifier(
        title,
        candidate_labels=FRAMES,
        multi_label=True,
        hypothesis_template="This news headline emphasizes {}."
    )

    scores = dict(zip(
        result["labels"],
        result["scores"]
    ))

    frames = [
        label
        for label, score in scores.items()
        if score >= threshold
    ]

    return {
        "frames": frames,
        "scores": scores
    }
"""

def detect_frames2(title):

    frames = [
        "Economy and finances",
        "Government",
        "Security and defense",
        "Commerce and trade",
        "Stories and entertainment",
        "Technology and AI",
        "Environment and climate change",
        "Geostrategy and war",
        "Sports",
        "Analysis and opinion"]

    result = classifier(
        title,
        candidate_labels=frames,
        multi_label=True
    )

    return result["labels"][0]

def detect_tone(title):

    frames = [
        "Economy and finances",
        "Government",
        "Security and defense",
        "Commerce and trade",
        "Stories and entertainment",
        "Technology and AI",
        "Environment and climate change",
        "Geostrategy and war",
        "Sports",
        "Analysis and opinion"]

    result = classifier(
        title,
        candidate_labels=frames,
        multi_label=True
    )

    return result["labels"][0]



#Arreglo Columna

#df.columns = ["title"]
#df = df.drop_duplicates(subset="title", keep="first")

#Analysis

df["frame_analysis"] = df["title"].apply(detect_frames2)

"""
df["frames"] = df["frame_analysis"].apply(
    lambda x: x["frames"]
)

df["frame_scores"] = df["frame_analysis"].apply(
    lambda x: x["scores"]
)
"""
# Resultado

frame_counts = (
    df["frame_analysis"]
    .explode()
    .dropna()
    .value_counts()
)

print(frame_counts)

#guardamos

df.to_csv("test_titles.csv", index=False, encoding="utf-8")
