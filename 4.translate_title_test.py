"""
Translates Texts and Titles to English
"""
""" Done """

import pandas as pd
import os
import requests
from dotenv import load_dotenv
import time
import requests

load_dotenv()

API_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT")
REGION = os.getenv("AZURE_TRANSLATOR_REGION")

def translate_text_azure(text, target="en"):

    if not isinstance(text, str) or not text.strip():
        return None

    url = ENDPOINT.rstrip("/") + "/translate"

    headers = {
        "Ocp-Apim-Subscription-Key": API_KEY,
        "Content-Type": "application/json"
    }

    if REGION:
        headers["Ocp-Apim-Subscription-Region"] = REGION

    params = {
        "api-version": "3.0",
        "to": target
    }

    for attempt in range(4):

        response = requests.post(
            url,
            params=params,
            headers=headers,
            json=[{"Text": text}],
            timeout=60
        )

        if response.status_code == 429:
            print("Azure error:", response.text)

            retry_after = response.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else 15 * (2 ** attempt)

            print(f"Waiting {wait} seconds...")
            time.sleep(wait)
            continue

        response.raise_for_status()

        return response.json()[0]["translations"][0]["text"]

    raise RuntimeError("Azure rate limit persists after retries")

def main():

    #Abrimos archivos
    df_texts = pd.read_csv("texts_to_translate.csv",
                            index_col=False,
                           header=0,
                           encoding= "utf-8")


    df_titles = pd.read_csv("titles_to_translate.csv",
                            index_col=False,
                            header=0,
                            encoding= "utf-8")

    print("=== TRANSLATE START ===")
    #Anadimos columnas traducidas
    translations = []

    for text in df_texts["body"]:

        translated = translate_text_azure(text)
        translations.append(translated)

        if isinstance(text, str):
            time.sleep(max(1, len(text) / 500))

    df_texts["body_en"] = translations

    #anadimos una columna de titulos traducidos
    df_titles["title_en"] = df_titles["title"].apply(translate_text_azure)
    df_texts = pd.merge(df_texts, df_titles[["title","title_en"]], on="title", how="inner")


    #Guardamos los datasets
    df_texts.to_csv("gold_texts.csv", index=False, encoding="utf-8")
    df_titles.to_csv("gold_titles.csv", index=False, encoding="utf-8")

    print("=== TRANSLATE COMPLETED ===")

if  __name__ == "__main__":
    main()



