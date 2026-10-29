import pandas as pd

""" DONE """
#FUNCIONES

# FRAGMENTACION
import pandas as pd

def fragment_text(column, max_words=100):

    fragments = []

    for article_id, text in column.dropna().items():

        if not isinstance(text, str) or not text.strip():
            continue

        words = text.split()

        for i in range(0, len(words), max_words):
            fragment = " ".join(words[i:i + max_words])

            if fragment.strip():
                fragments.append({
                    "article_id": article_id,
                    "fragment_id": len(fragments),
                    "fragment": fragment
                })

    return pd.DataFrame(
        fragments,
        columns=["article_id", "fragment_id", "fragment"]
    )

# ELIMINAR DUPLICADOS SEMANTICOS
from sklearn.metrics.pairwise import cosine_similarity

def remove_semantic_duplicates(df, threshold=0.95):

    df = df.copy().reset_index(drop=True)

    embeddings = np.stack([
        np.fromstring(x.strip("[]"), sep=" ")
        if isinstance(x, str)
        else np.asarray(x, dtype=float)
        for x in df["embedding"]
    ])

    similarity = cosine_similarity(embeddings)

    keep = []

    for i in range(len(df)):

        if not keep or np.max(similarity[i, keep]) < threshold:
            keep.append(i)

    return df.iloc[keep].reset_index(drop=True)

# SE CREAN LOS EMBEDDINGS

from sentence_transformers import SentenceTransformer

# Cargar el modelo una sola vez
model = SentenceTransformer("all-mpnet-base-v2")

def generate_embeddings(df):

    df = df.copy()

    embeddings = model.encode(
        df["fragment"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    df["embedding"] = list(embeddings)

    return df

#CLUSTERIZATION
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize

def cluster_fragments(df, min_cluster_size=5, min_samples=2):

    df = df.copy()

    if len(df) < min_cluster_size:
        df["cluster"] = -1
        df["cluster_probability"] = 0.0
        return df

    # Convertir embeddings a números
    embeddings = np.stack([
        np.fromstring(x.strip("[]"), sep=" ")
        if isinstance(x, str)
        else np.asarray(x, dtype=float)
        for x in df["embedding"]
    ])

    embeddings = normalize(embeddings)

    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean"
    )

    df["cluster"] = model.fit_predict(embeddings)
    df["cluster_probability"] = model.probabilities_

    return df
#CLUSTERIZACION 2

import numpy as np
import umap
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize


def cluster_fragments2(df):

    df = df.copy().reset_index(drop=True)

    # Extract embeddings
    embeddings = np.stack(df["embedding"].values)

    # Normalize embeddings
    embeddings = normalize(embeddings)

    # Dimensionality reduction
    reducer = umap.UMAP(
        n_neighbors=8,
        n_components=11,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    reduced_embeddings = reducer.fit_transform(embeddings)

    # Clustering
    model = HDBSCAN(
        min_cluster_size=3,
        min_samples=2,
        metric="euclidean"
    )

    df["cluster"] = model.fit_predict(reduced_embeddings)

    df["cluster_probability"] = model.probabilities_

    return df

# SAVE CLUSTER AS TXT
def save_clusters(df, filename="clusters.txt"):

    with open(filename, "w", encoding="utf-8") as file:

        for cluster_id, group in df.groupby("cluster"):

            file.write(f"\n{'=' * 60}\n")
            file.write(
                f"CLUSTER ID: {cluster_id} | {len(group)} fragmentos\n"
            )
            file.write(f"{'=' * 60}\n")

            for _, row in group.iterrows():

                file.write(
                    f"FRAGMENT ID: {row['fragment_id']}\n"
                )

                file.write(f"{row['fragment']}\n")

    print(f"Clusters guardados en {filename}")

def main():

    #FUNCIONA

    df = pd.read_csv("gold_texts.csv",
                     encoding="utf-8")

    #Eliminamos Duplicados

    df = df.drop_duplicates(subset=["body_en"],
                            keep="first")

    #Fragmentamos en un nuevo df

    df_frag = fragment_text(df["body_en"])

    print("NUMBER OF FRAGMENTS",df_frag.shape)

    df_frag.to_csv("fragments_test.csv", index=False, encoding="utf-8")

    print("=== FRAGMENTATION FINISH, EMBEDDINGS START ===")


    df_embeddings = generate_embeddings(df_frag)

    print("Dimensiones")

    print(df.shape)
    print(df_frag.shape)
    print(df_embeddings.shape)

    df_embeddings.to_csv("embbedings_test.csv", index=False, encoding="utf-8")

    print("EMBEDDINGS COMPLETED, CLUSTERIZATION STARTS")

    #COMIENZA CLUSTERIZACION DE LOS EMBEDDINGS

    """
    #LIMPIAMOS DUPLICADOS SEMANTICOS
    
    df_clean = remove_semantic_duplicates(df_embeddings,
                                           threshold=0.95)
    
    print("SEMANTIC DUPLICATE CLEANING COMPLETED")
    print(df_embeddings.shape)
    print(df_clean.shape)
    """

    #CLUSTERIZATION
    df_clusters = cluster_fragments2(df_embeddings)

    print(df_clusters["cluster"].value_counts().sort_index())

    #Ensenar el texto de los clusters
    print("SHAPE CONTROL")
    print(df.shape)
    print(df_frag.shape)
    print(df_embeddings.shape)
    print(df_clusters.shape)

    print("SAVING EMBEDDINGS IN clusters_test.csv AND clusters.txt")
    df_clusters.to_csv("clusters_test.csv", index=False, encoding="utf-8")
    save_clusters(df_clusters, "clusters.txt")

if __name__ == "__main__":
    main()
