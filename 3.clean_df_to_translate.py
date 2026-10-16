import pandas as pd

# Abrimos el df
df = pd.read_csv("texts.csv",
                 header=0,
                 encoding="utf-8")

#Guardamos los titulos
df_titles = df["title"].copy()
df_titles.to_csv("titles_to_translate.csv", index=False, encoding="utf-8")

#limpiamos columnas que no necesitamos

df.drop(columns = ["extraction_method",
                      "extraction_status",
                      "url",
                      "language"],inplace=True)

#limpiamos filas sin texto de noticia
df = df.dropna(subset=["body"])

#limpiamos filas con un texto mayo a 11k caracteres

df = df[df["body"].str.len()<11000]

#limpiamos filas con texto sin sentido

trash=["°","³"]

df = df[~df['body'].fillna('').str.contains('|'.join(trash), case=False, regex=True)]

# reiniciamos el indice

df = df.reset_index(drop=True)

# Guardamos

df.to_csv("texts_to_translate.csv", index=False, encoding="utf-8")
