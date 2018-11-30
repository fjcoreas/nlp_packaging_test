import spacy
import seaborn as sns
from data import t0, t1, t2, t3, t4, t5, t6
from processing import tf_idf_scores
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

doc0 = nlp(t0)
doc1 = nlp(t1)
doc2 = nlp(t2)
doc3 = nlp(t3)
doc4 = nlp(t4)
doc5 = nlp(t5)
doc6 = nlp(t6)

df = tf_idf_scores([doc0,doc1,doc2,doc3,doc4,doc5,doc6])

df_norm_col=(df-df.mean())/df.std()
sns.heatmap(df_norm_col, cmap='BuPu',vmin=0, vmax=1.5)
plt.show()
plt.savefig('tf_idf_scores.png')



