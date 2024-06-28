import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('ggplot')
import nltk
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