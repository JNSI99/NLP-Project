from transformers import pipeline
import pandas as pd
import spacy

""" DONE """

# Clasificador

classifier = pipeline(
    task="zero-shot-classification",
    model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    device=-1
)

# Framing detection
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
        "Analysis and opinion",
        "Other"]

    result = classifier(
        title,
        candidate_labels=frames,
        multi_label=True
    )

    return result["labels"][0]

#tone detection
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

#ROLE EXTRACTION PIPELINE

nlp = spacy.load("en_core_web_sm")
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

# ##### MAIN FUNCTION #####
def main():

        #open the document
        df = pd.read_csv("gold_titles.csv",
                         encoding="utf-8")

        #Arreglo Columna DROP DUPLICATES
        df = df.drop_duplicates(subset="title", keep="first")

        #Analysis
        print("=== FRAME-TONE ANALYSIS START ===")
        df["frame_analysis"] = df["title_en"].apply(detect_frames2)
        df["tone"] = df["title_en"].apply(detect_tone)

        # Resultados
        frame_counts = (
            df["frame_analysis"]
            .explode()
            .dropna()
            .value_counts()
        )

        tone_counts = (
            df["tone"]
            .explode()
            .dropna()
            .value_counts()
        )

        print(frame_counts)
        print(tone_counts)

        #guardamos
        print("=== SAVING FRAME ANALYSIS IN test_titles.csv ===")
        df.to_csv("test_titles.csv", index=False, encoding="utf-8")

        print("=== ACTOR-ROLE ANALYSIS START ===")

        df_actors = actor_framing(df, title_col="title_en")

        print("=== SAVING ACTOR-ROLE ANALYSIS IN final_test.csv ===")
        df_actors.to_csv("final_test.csv", index=False, encoding="utf-8")

        role_counts = (
            df_actors["actor_role"]
            .explode()
            .dropna()
            .value_counts()
        )

        print(role_counts)

if __name__ == "__main__":
    main()



