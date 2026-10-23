from transformers import pipeline
import pandas as pd

#open the document

df = pd.read_csv("test_titles.csv",
                 encoding="utf-8")

#Clasificador

classifier = pipeline(
    task="zero-shot-classification",
    model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    device=-1
)

#Classificator
def detect_tone(title):

    tones = [
        "Optimistic",
        "Concerned",
        "Critical",
        "Neutral",
        "Alarmist",
        "Reassuring"
    ]

    result = classifier(
        title,
        candidate_labels=tones,
        multi_label=True
    )

    return result["labels"][0]


df['tone'] = df['title'].apply(detect_tone)

df.to_csv("test_titles.csv", index=False, encoding="utf-8")