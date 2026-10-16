import pandas as pd
from newspaper import languages

df = pd.read_csv("texts.csv",
                 index_col=False,
                 header=0,
                 encoding="utf-8")

# Preselection of the texts

lens = df['body'].str.len()
lens = lens[lens < 11000]

df = df[ df['body'].str.len() < 11000 ]

#Tranlsations Function

import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT")
REGION = os.getenv("AZURE_TRANSLATOR_REGION")


def translate_text(text, target="en", source="auto"):

    if not isinstance(text, str) or not text.strip():
        return None

    if len(text) > 50000:
        raise ValueError("Text exceeds Azure's 50,000-character limit")

    if not API_KEY or not ENDPOINT:
        raise ValueError("Missing Azure Translator credentials")

    url = ENDPOINT.rstrip("/") + "/translate"

    params = {
        "api-version": "3.0",
        "to": target
    }

    # Si no indicamos idioma, Azure lo detecta
    if source != "auto":
        params["from"] = source

    headers = {
        "Ocp-Apim-Subscription-Key": API_KEY,
        "Content-Type": "application/json"
    }

    if REGION:
        headers["Ocp-Apim-Subscription-Region"] = REGION

    response = requests.post(
        url,
        params=params,
        headers=headers,
        json=[{"Text": text}],
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    return data[0]["translations"][0]["text"]

#Test dataframe
df_test = df.iloc[[4,15,44,63,32]].copy()

#Translation
df_test["body_en"] = df_test["body"].apply(translate_text)




