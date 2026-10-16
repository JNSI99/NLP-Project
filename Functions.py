
# Funcion de traducir el input a buscar

def translate_input(text):
    # gets a string in english and returns
    # a tuple with the translatations in different languages

    from deep_translator import MyMemoryTranslator

    # Translations

    t1 = MyMemoryTranslator(source='en-US',target='ru-RU').translate(text)
    t2 = MyMemoryTranslator(source='en-US',target='zh-CN').translate(text)
    t3 = MyMemoryTranslator(source='en-US',target='es-ES').translate(text)
    t4 = MyMemoryTranslator(source='en-US',target='ar-SA').translate(text)
    t5 = MyMemoryTranslator(source='en-US',target='sw-KE').translate(text)

    return [text,t1,t2,t3,t4,t5]

# Funcion de buscar en la web unn input

regions = ["us-en",
           "ru-ru",
           "cn-zh",
           "es-es",
           "xa-ar",
           "wt-wt" ]

# URL SEARCH Functions

def search_news(topic, max_results, max_sources=5, region="us-en"):
    import pandas as pd
    from ddgs import DDGS
    from urllib.parse import urlparse
    from collections import defaultdict

    # Buscamos más candidatos de los necesarios
    candidates = max_results * max_sources * 5

    with DDGS() as ddgs:
        results = list(ddgs.news(
            query=topic,
            max_results=candidates,
            region=region
        ))

    # Agrupar artículos por fuente
    sources = defaultdict(list)
    seen_urls = set()

    for article in results:

        title = article.get("title")
        url = article.get("url")

        if not title or not url:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        # Identificar fuente mediante dominio
        domain = urlparse(url).netloc.lower()
        domain = domain.removeprefix("www.")

        if not domain:
            continue

        # Máximo max_results artículos por fuente
        if len(sources[domain]) < max_results:
            sources[domain].append({
                "title": title,
                "url": url,
                "source": domain
            })

    # Seleccionar las fuentes con más artículos
    selected_sources = sorted(
        sources,
        key=lambda source: len(sources[source]),
        reverse=True
    )[:max_sources]

    articles = []

    for source in selected_sources:
        articles.extend(sources[source])

    return pd.DataFrame(
        articles,
        columns=["title", "url", "source"]
    )

def search_newsv2(topic, max_results, max_sources=5, region="us-en"):
    import pandas as pd
    from ddgs import DDGS
    from urllib.parse import urlparse
    from collections import defaultdict

    # Buscamos más candidatos de los necesarios
    candidates = max_results * max_sources * 5

    results = []

    with DDGS() as ddgs:

        # 1. Intentamos buscar noticias
        try:
            results = list(ddgs.news(
                query=topic,
                max_results=candidates,
                region=region
            ))

            print(f"DDGS News: {len(results)} resultados")

        except Exception as e:
            print(f"DDGS News error: {e}")

        # 2. Si no hay resultados, usamos DDGS Text
        if not results:
            print("Trying DDGS Text...")

            try:
                results = list(ddgs.text(
                    query=topic,
                    max_results=candidates,
                    region=region
                ))

                print(f"DDGS Text: {len(results)} resultados")

            except Exception as e:
                print(f"DDGS Text error: {e}")

    # Agrupar artículos por fuente
    sources = defaultdict(list)
    seen_urls = set()

    for article in results:

        title = article.get("title")

        # DDGS News utiliza "url"
        # DDGS Text utiliza "href"
        url = article.get("url") or article.get("href")

        if not title or not url:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        # Identificar fuente mediante dominio
        domain = urlparse(url).netloc.lower()
        domain = domain.removeprefix("www.")

        if not domain:
            continue

        # Máximo max_results artículos por fuente
        if len(sources[domain]) < max_results:
            sources[domain].append({
                "title": title,
                "url": url,
                "source": domain
            })

    # Seleccionar las fuentes con más artículos
    selected_sources = sorted(
        sources,
        key=lambda source: len(sources[source]),
        reverse=True
    )[:max_sources]

    articles = []

    for source in selected_sources:
        articles.extend(sources[source])

    return pd.DataFrame(
        articles,
        columns=["title", "url", "source"]
    )

# Funcion de obtener el texto dentro del enlace

#!!!Valida en msn
def extract_page_text1(url):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=False
        )

        try:
            page = browser.new_page()

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=30000
            )

            # Buscar el cuerpo de la noticia
            article = page.locator(
                '[data-test-id="test-article"] body'
            ).first

            article.wait_for(
                state="attached",
                timeout=20000
            )

            text = article.inner_text(
                timeout=10000
            )

            return text

        finally:
            browser.close()


#!!!Pipeline exitoso para extraccion de texto

#Librerias
import pandas as pd
import requests
import trafilatura
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

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

#Text Translation

# (Azure)

#Text Analysis


