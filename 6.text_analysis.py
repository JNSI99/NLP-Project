import pandas as pd


#FUNCIONA

df = pd.read_csv("gold_texts.csv",
                 encoding="utf-8",
                 )
#Eliminamos Duplicados

df = df.drop_duplicates(subset=["body_en"],
                        keep="first")

# Fragmentamos

def fragment_text(column, max_words=100):

    fragments = []

    for text in column.dropna():

        if not isinstance(text, str) or not text.strip():
            continue

        words = text.split()

        for i in range(0, len(words), max_words):
            fragment = " ".join(words[i:i + max_words])

            if fragment.strip():
                fragments.append(fragment)

    return pd.DataFrame({"fragment": fragments})

df_frag = fragment_text(df["body_en"])

import numpy as np
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

#limpieza de fragmentos

df_frag.to_csv("fragments_test.csv", index=False, encoding="utf-8")

print("Fragmentacion completada comienzas los embeddings")

#Creamos los embeddings
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

df_embeddings = generate_embeddings(df_frag)

print("Dimensiones")

print(df.shape)
print(df_frag.shape)
print(df_embeddings.shape)

df_embeddings.to_csv("embbedings_test.csv", index=False, encoding="utf-8")

print("Embeddings completados comienza la clustrizacion")

#COMIENZA CLUSTERIZACION DE LOS EMBEDDINGS
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(df):

    embeddings = np.stack([
        np.fromstring(
            x.strip("[]"),
            sep=" "
        ) if isinstance(x, str) else np.asarray(x, dtype=float)
        for x in df["embedding"]
    ])

    similarity = cosine_similarity(embeddings)

    return pd.DataFrame(similarity)

df_similarity = calculate_similarity(df)

print("df cosine similarity", df_similarity.shape)


Representacion de cosine similarity
import matplotlib.pyplot as plt

#Funcion para ver graficamente cosine similarity
def plot_similarity(df_similarity):

    plt.figure(figsize=(10, 8))

    plt.imshow(
        df_similarity.values,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.colorbar(label="Cosine Similarity")
    plt.title("Semantic Similarity Between News Fragments")
    plt.xlabel("Fragment ID")
    plt.ylabel("Fragment ID")

    plt.tight_layout()
    plt.show()

plot_similarity(df_similarity)
"""

#Hdbscan

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize

#limpiamos

df_clean = remove_semantic_duplicates(df_embeddings,
                                       threshold=0.95)

print("Limpieza duplicados completada")
print(df_embeddings.shape)
print(df_clean.shape)

#Clusterizacion de los embeddings

def cluster_fragments(df, min_cluster_size=2, min_samples=2):

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


df_clusters = cluster_fragments(df_embeddings)

print(df_clusters["cluster"].value_counts().sort_index())

#Ensenar el texto de los clusters
def show_clusters(df):

    for cluster_id, group in df.groupby("cluster"):

        print(f"\n{'=' * 60}")
        print(f"CLUSTER {cluster_id} | {len(group)} fragmentos")
        print(f"{'=' * 60}")

        for i, fragment in enumerate(group["fragment"], 1):
            print(f"\nFragmento {i}:")
            print(fragment)

def show_cluster(df, cluster_id):

    fragments = df[df["cluster"] == cluster_id]

    for i, fragment in enumerate(fragments["fragment"], 1):
        print(f"\n--- Fragmento {i} ---")
        print(fragment)

def save_clusters(df, filename="clusters.txt"):

    with open(filename, "w", encoding="utf-8") as file:

        for cluster_id, group in df.groupby("cluster"):

            file.write(f"\n{'=' * 60}\n")
            file.write(
                f"CLUSTER {cluster_id} | {len(group)} fragmentos\n"
            )
            file.write(f"{'=' * 60}\n")

            for i, fragment in enumerate(group["fragment"], 1):
                file.write(f"\nFragmento {i}:\n")
                file.write(f"{fragment}\n")

    print(f"Clusters guardados en {filename}")

print("dimensiones 2")

print(df.shape)
print(df_frag.shape)
print(df_embeddings.shape)
print(df_clusters.shape)

df_clusters.to_csv("clusters_test.csv", index=False, encoding="utf-8")

save_clusters(df_clusters, "clusters.txt")
