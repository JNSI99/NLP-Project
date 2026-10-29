import pandas as pd

"""DONE"""

def main():

        # Abrimos el df

        df = pd.read_csv("texts.csv",
                         header=0,
                         encoding="utf-8")

        #Borramos columnas que no necesitamos
        df.drop(columns = ["extraction_method",
                              "extraction_status",
                              "url","source"],inplace=True)

        #Guardamos los titulos

        df_titles = df[["language","title","article_id"]].copy()
        print("=== TITLES SAVE IN titles_to_translate.csv ===")
        df_titles.to_csv("titles_to_translate.csv", index=False, encoding="utf-8")


        #limpiamos filas sin texto de noticias

        df_bodies = df[["title","language","body","article_id"]].copy()
        df_bodies = df_bodies.dropna(subset=["body"])

        #limpiamos filas con un texto mayo a 11k caracteres

        df_bodies = df_bodies[df_bodies["body"].str.len()<11000]

        #limpiamos filas con texto sin sentido (!!REVISAR!!)

        trash=["³","°"]
        df_bodies = df_bodies[~df_bodies['body'].fillna('').str.contains('|'.join(trash), case=False, regex=True)]

        #GUARDAMOS BODIES
        print("=== TEXTS SAVES IN texts_to_translate.csv ===")
        df_bodies.to_csv("texts_to_translate.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
        main()
