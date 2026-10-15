
import Functions as xd

text = "Climate change"

titular = xd.translate_input(text)
print(titular)

df = xd.search_newsv2(topic = titular[4],
                 max_results=20,
                 max_sources=10,
                 region="xa-ar")

print(df)
print(df.shape)
