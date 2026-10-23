import pandas as pd
import spacy
from transformers import pipeline
import pandas as pd

# Abrimos df

df = pd.read_csv("test_titles.csv",encoding="utf-8")

df = df.iloc[43:]

# Cargar modelos una sola vez
nlp = spacy.load("en_core_web_sm")

classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    device=-1
)

# 1. Identificar actores
def extract_actors(title):

    doc = nlp(title)

    return list(dict.fromkeys(
        ent.text
        for ent in doc.ents
        if ent.label_ in ["PERSON", "ORG", "GPE"]
    ))

# 2. Identificar el rol narrativo de cada actor
def detect_actor_role(title, actor):

    ACTOR_ROLES = [
        "Promoting cooperation",
        "Solving a problem",
        "Creating a threat",
        "Responsible for a problem",
        "Affected by an event",
        "Facing criticism",
        "Neutral participant"
    ]

    result = classifier(
        f"Headline: {title}\nActor: {actor}",
        candidate_labels=ACTOR_ROLES,
        hypothesis_template="The actor is portrayed as {}.",
        multi_label=False
    )

    return result["labels"][0]

# 3. Procesar el DataFrame
def actor_framing(df, title_col="title_en"):

    df_result = df.copy()

    # Extraer actores
    df_result["actor"] = df_result[title_col].apply(
        lambda x: extract_actors(x)
        if isinstance(x, str) else []
    )

    # Una fila por actor
    df_result = df_result.explode("actor")

    # Eliminar filas sin actores
    df_result = df_result.dropna(subset=["actor"]).copy()

    # Clasificar cada actor
    df_result["actor_role"] = df_result.apply(
        lambda row: detect_actor_role(
            row[title_col],
            row["actor"]
        ),
        axis=1
    )

    return df_result.reset_index(drop=True)

df_actors = actor_framing(df, title_col="title")

df_actors.to_csv("final_test.csv", index=False, encoding="utf-8")
