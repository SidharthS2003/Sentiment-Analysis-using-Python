import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('ggplot')
import nltk
df=pd.read_csv("/kaggle/input/amazon-fine-reviews/Reviews.csv")
df=df.head(500)
ax=df['Score'].value_counts().sort_index().plot(kind='bar',title="Count of Reviews by stars")
ax.set_xlabel("REVIEW STARS")
ax.set_ylabel("PEOPLE COUNT")
example=df['Text'][50]
tokens=nltk.word_tokenize(example)
tagged=nltk.pos_tag(tokens)
tagged[:10]
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
sia.polarity_scores(example)
res={}
for i,row in df.iterrows():
    text=row['Text']
    id=row["Id"]
    res[id]=sia.polarity_scores(text)
    
vaders=pd.DataFrame(res).T
vaders=vaders.reset_index().rename(columns={'index':'Id'})
vaders=vaders.merge(df,how='left')
fig,axs=plt.subplots(1,3,figsize=(15,5))
ax = sns.barplot(data=vaders, x='Score', y='pos',ax=axs[0])
ax = sns.barplot(data=vaders, x='Score', y='neu',ax=axs[1])
ax = sns.barplot(data=vaders, x='Score', y='neg',ax=axs[2])
axs[0].set_title("Positive")
axs[1].set_title("Neutral")
axs[2].set_title("Negative")
