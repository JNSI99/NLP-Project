# Url extraction script
import Functions as xd
import pandas as pd

"""DONE"""

# Example
text = "Xi Jinping official state visit to the US September 23 to 25"


def main(text):

    # Text translation
     translation = xd.translate_input(text)

     print("====== TRANSLATION COMPLETED =======")

    #Regions

     regions = ["us-en",
           "ru-ru",
           "wt-wt",
           "es-es",
           "wt-wt"]

     languages = ["en",
             "ru",
             "cn",
             "es",
             "ar"]



     #Text Search in different languages

     dfs = []

     for i in range(0,len(regions)):

          df = xd.search_newsv2(topic = translation[i],
                        max_results=10,
                        max_sources=10,
                        region = regions[i])

          df["language"] = languages[i]

          dfs.append(df)

     print("==== URL SEARCH COMPLETED ====")

#df concat

     df_final = pd.concat([dfs[0],dfs[1],dfs[2],dfs[3],dfs[4]],ignore_index=True)
     print("Final df size", df_final.shape)

#Save the urls

     df_final.to_csv("urls.csv",index=False,encoding="utf-8")
     print("===== URLS SAVED IN urls.csv =====")

if __name__ == "__main__":
     main(text)


