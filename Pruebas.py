import pandas as pd

df1 = pd.read_csv("urls.csv",encoding = "utf-8")
print("urls.csv")
print(df1.shape)
print(df1.columns)

df2 = pd.read_csv("titles_to_translate.csv",encoding = "utf-8")
print("titles_to_translate.csv")
print(df2.shape)
print(df2.columns)


df3 = pd.read_csv("texts_to_translate.csv",encoding = "utf-8")
print("texts_to_translate.csv")
print(df3.shape)
print(df3.columns)

df4 = pd.read_csv("texts.csv",encoding = "utf-8")
print("texts.csv")
print(df4.shape)
print(df4.columns)

df5 = pd.read_csv("test_titles.csv",encoding = "utf-8")
print("test_titles.csv")
print(df5.shape)
print(df5.columns)

df6 = pd.read_csv("gold_titles.csv",encoding = "utf-8")
print("gold_titles.csv")
print(df6.shape)
print(df6.columns)

df7 = pd.read_csv("gold_texts.csv",encoding = "utf-8")
print("gold_texts.csv")
print(df7.shape)
print(df7.columns)

df8 = pd.read_csv("fragments_test.csv",encoding = "utf-8")
print("fragments_test.csv")
print(df8.shape)
print(df8.columns)

df9 = pd.read_csv("final_test.csv",encoding = "utf-8")
print("final_test.csv")
print(df9.columns)

df10 = pd.read_csv("embbedings_test.csv",encoding = "utf-8")
print("embbedings_test.csv")
print(df10.columns)

df11 = pd.read_csv("clusters_test.csv",encoding = "utf-8")
print("clusters_test.csv")
print(df11.columns)
