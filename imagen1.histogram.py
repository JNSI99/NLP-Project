import pandas as pd

df = pd.read_csv("texts.csv",
                 index_col=False,
                 header=0,
                 encoding="utf-8")

lens = df['body'].str.len()
lens = lens[lens < 11000]

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.hist(lens,bins=40,color="blue")

plt.title("Histogram")
plt.xlabel("Length")
plt.ylabel("Frequency")

plt.grid(True)

plt.show()