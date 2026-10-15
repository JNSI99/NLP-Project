
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

