from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
import pandas as pd
from utils import *

# reference
# https://towardsdatascience.com/nlp-with-lda-latent-dirichlet-allocation-and-text-clustering-to-improve-classification-97688c23d98
# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
# https://www.kaggle.com/code/ykhorramz/lda-and-t-sne-interactive-visualization/notebook
# https://towardsdatascience.com/text-analysis-basics-in-python-443282942ec5

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += ", ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)

def lda_model(df, col_in, sw_in, min_in, max_in, num_of_topics, ntop_w, sw=[]):
  """
  Input:
    df_col_in: text corpus,
    sw_in: vectorizer type,
    min_in: min ngram,
    max_in: max ngram,
    num_of_topics: number of topics to identify,
    ntop_w: top n words,
    sw: additonal stop words to remove
  Ouput:
    Topics and top ranked words associated with the topic
  """
  df["comment_clean"] = df[col_in].apply(clean_text).apply(rem_sw)

  stoplist = stopwords.words('english') + ['www', 'http', 'https', 'com', 'html', 'index'] + ["trump", "politics"] + sw
  # Vecotrize tokens
  if sw_in == "tf-idf":
    cv = TfidfVectorizer(stop_words=stoplist, ngram_range=(min_in, max_in))
  else:
    cv = CountVectorizer(stop_words=stoplist, ngram_range=(min_in, max_in))

  # Train model
  lda_model = LDA(n_components=num_of_topics)
  pipe = make_pipeline(cv, lda_model)
  pipe.fit(df.comment_clean)

  print_top_words(lda_model, cv.get_feature_names(), n_top_words=ntop_w)

  return lda_model

data = pd.read_csv("culture.csv")
model = lda_model(data, comment, sw_in="tf-idf", min_in=2, max_in=3, num_of_topics=2, ntop_w=5)
