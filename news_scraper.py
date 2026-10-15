"""
GlobalPulse NLP - multilingual news scraper.

Searches GDELT for articles in English, Spanish, Chinese, Arabic, and Russian,
then attempts to extract the article text with Trafilatura.

Important:
- GDELT search terms should be provided in the language being searched when possible.
- Pass translated/localized search terms with `queries_by_language`.
- Search results and full-text extraction are not guaranteed for every publisher.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping, Sequence
from urllib.parse import urlparse

import pandas as pd
import requests
import trafilatura
from deep_translator import GoogleTranslator


GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

LANGUAGES = {
    "en": {"name": "English", "gdelt_name": "english"},
    "es": {"name": "Spanish", "gdelt_name": "spanish"},
    "zh": {"name": "Chinese", "gdelt_name": "chinese"},
    "ar": {"name": "Arabic", "gdelt_name": "arabic"},
    "ru": {"name": "Russian", "gdelt_name": "russian"},
}

REQUEST_TIMEOUT = 25
USER_AGENT = (
    "GlobalPulseNLP/0.1 "
    "(multilingual news research; respectful rate-limited access)"
)


def translate_topic(topic: str, target_language: str) -> str:
    """Translate a topic automatically into one of the supported languages."""
    if target_language == "en":
        return topic.strip()
    try:
        translated = GoogleTranslator(source="auto", target=target_language).translate(
            topic.strip()
        )
        return translated or topic.strip()
    except Exception:
        # If translation fails, keep the scraper running with the original topic.
        return topic.strip()


def search_news(
    query: str,
    language: str,
    max_results: int = 10,
    timespan: str = "7d",
) -> list[dict]:
    """Search GDELT for articles matching a query in one language."""
    if language not in LANGUAGES:
        raise ValueError(
            f"Unsupported language {language!r}. Choose from {list(LANGUAGES)}."
        )
    if not query or not query.strip():
        raise ValueError("query cannot be empty.")
    if not 1 <= max_results <= 250:
        raise ValueError("max_results must be between 1 and 250.")

    # GDELT supports language filtering through the source-language operator.
    # The query text itself should be localized for best cross-lingual recall.
    gdelt_language = LANGUAGES[language]["gdelt_name"]
    params = {
        "query": f'("{query.strip()}") sourcelang:{gdelt_language}',
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_results,
        "sort": "DateDesc",
        "timespan": timespan,
    }

    response = requests.get(
        GDELT_DOC_API,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"GDELT returned a non-JSON response for language {language!r}."
        ) from exc

    articles = payload.get("articles", [])
    rows = []

    for article in articles:
        url = article.get("url")
        if not url:
            continue

        rows.append(
            {
                "query": query.strip(),
                "language": language,
                "language_name": LANGUAGES[language]["name"],
                "title": article.get("title"),
                "source": article.get("domain")
                or urlparse(url).netloc,
                "published_at": article.get("seendate"),
                "url": url,
                "description": None,
                "text": None,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "status": "discovered",
            }
        )

    return rows


def extract_article(url: str) -> dict:
    """Attempt to extract an article's main text and metadata."""
    if not url or not url.startswith(("http://", "https://")):
        return {"text": None, "extracted_title": None, "status": "invalid_url"}

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return {"text": None, "extracted_title": None, "status": "download_failed"}

    extracted = trafilatura.extract(
        downloaded,
        url=url,
        output_format="json",
        with_metadata=True,
        include_comments=False,
    )
    if not extracted:
        return {"text": None, "extracted_title": None, "status": "extraction_failed"}

    import json

    try:
        metadata = json.loads(extracted)
    except (TypeError, ValueError):
        return {"text": extracted, "extracted_title": None, "status": "extracted"}

    text = metadata.get("text")
    return {
        "text": text,
        "extracted_title": metadata.get("title"),
        "status": "extracted" if text else "extraction_failed",
    }


def scrape_news(topic: str, max_results: int = 10) -> pd.DataFrame:
    """
    Search all five configured languages using automatic topic translation.

    Public parameters:
        topic: topic to search for
        max_results: maximum number of results per language
    """
    if not topic or not topic.strip():
        raise ValueError("topic cannot be empty.")
    if not 1 <= max_results <= 250:
        raise ValueError("max_results must be between 1 and 250.")

    rows: list[dict] = []
    for language in ("en", "es", "zh", "ar", "ru"):
        localized_query = translate_topic(topic.strip(), language)
        try:
            language_rows = search_news(
                query=localized_query,
                language=language,
                max_results=max_results,
                timespan="7d",
            )
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            rows.append({
                "query": localized_query,
                "language": language,
                "language_name": LANGUAGES[language]["name"],
                "title": None,
                "source": None,
                "published_at": None,
                "url": None,
                "description": None,
                "text": None,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "status": f"search_error: {type(exc).__name__}: {exc}",
            })
            continue

        for row in language_rows:
            try:
                extracted = extract_article(row["url"])
                row["text"] = extracted["text"]
                row["status"] = extracted["status"]
                if extracted["extracted_title"] and not row["title"]:
                    row["title"] = extracted["extracted_title"]
            except Exception as exc:
                row["status"] = f"extraction_error: {type(exc).__name__}"
            rows.append(row)

    columns = [
        "query", "language", "language_name", "title", "source",
        "published_at", "url", "description", "text", "scraped_at", "status",
    ]
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    return df


def save_results(df: pd.DataFrame, path: str = "data/articles.csv") -> None:
    """Save results to CSV, creating the parent directory if needed."""
    from pathlib import Path

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    results = scrape_news(topic="climate change", max_results=10)
    print(results[["language", "title", "source", "status", "url"]].to_string(index=False))
    save_results(results)
    print(f"\\nSaved {len(results)} rows to data/articles.csv")
