import pandas as pd
import requests
import trafilatura
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

"""DONE"""

# --------------------------------------------------
# 1. Extracción rápida utilizando Trafilatura
# --------------------------------------------------

def extract_with_trafilatura(url):

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15
        )

        response.raise_for_status()

        text = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=False,
            favor_precision=True
        )

        return text if text and text.strip() else None

    except Exception:
        return None


# --------------------------------------------------
# 2. Extracción utilizando Playwright
# --------------------------------------------------

def extract_with_playwright(page, url):

    page.goto(
        url,
        wait_until="domcontentloaded",
        timeout=30000
    )

    domain = urlparse(url).netloc.lower()

    # Caso especial: MSN
    if domain == "msn.com" or domain.endswith(".msn.com"):

        try:
            article = page.locator(
                '[data-test-id="test-article"] body'
            ).first

            article.wait_for(
                state="attached",
                timeout=15000
            )

            text = article.inner_text(
                timeout=10000
            )

            if text and text.strip():
                return text, "playwright_msn"

        except PlaywrightTimeout:
            pass

    # Alternativa general: HTML renderizado
    try:
        page.locator("body").first.wait_for(
            state="attached",
            timeout=5000
        )
    except PlaywrightTimeout:
        pass

    html = page.content()

    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False
    )

    if text and text.strip():
        return text, "playwright_trafilatura"

    return None, None


# --------------------------------------------------
# 3. Procesar un DataFrame de noticias
# --------------------------------------------------

def extract_articles(df):

    df = df.copy()

    required_columns = {"url"}

    if not required_columns.issubset(df.columns):
        raise ValueError("DataFrame must contain a 'url' column")

    bodies = []
    statuses = []
    methods = []

    # Abrimos Chromium una sola vez
    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        try:
            total = len(df)

            for i, url in enumerate(df["url"], start=1):

                print(f"Processing {i}/{total}: {url}")

                text = None
                method = None
                status = "failed"

                if not isinstance(url, str) or not url.startswith(
                    ("http://", "https://")
                ):
                    bodies.append(None)
                    statuses.append("invalid_url")
                    methods.append(None)
                    continue

                # Primer intento: Trafilatura
                text = extract_with_trafilatura(url)

                if text:
                    method = "trafilatura"

                # Segundo intento: Playwright
                else:
                    page = None

                    try:
                        page = browser.new_page()

                        text, method = extract_with_playwright(
                            page, url
                        )

                    except Exception as e:
                        print(f"Playwright error: {e}")

                    finally:
                        if page is not None:
                            page.close()

                # Validación básica
                if text and len(text.strip()) >= 100:
                    status = "success"
                else:
                    status = "failed"
                    text = None
                    method = None

                bodies.append(text)
                statuses.append(status)
                methods.append(method)

                print(f"Status: {status} | Method: {method}")

        finally:
            browser.close()

    # Añadimos las columnas al DataFrame
    df["body"] = bodies
    df["extraction_status"] = statuses
    df["extraction_method"] = methods

    return df

def main():

    df = pd.read_csv("urls.csv",
                     index_col=False,
                     header=0,
                     encoding="utf-8")

    print("=== TEXT EXTRACTION ===")

    df_final = extract_articles(df)

    print("=== SAVING TEXTS IN texts.csv ===")

    df_final.to_csv("texts.csv",index=False,encoding="utf-8")

if __name__ == "__main__":
    main()


