from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
import pandas as pd

# reference
# https://towardsdatascience.com/nlp-with-lda-latent-dirichlet-allocation-and-text-clustering-to-improve-classification-97688c23d98
# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
# https://www.kaggle.com/code/ykhorramz/lda-and-t-sne-interactive-visualization/notebook

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += ", ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)

def lda_model(df_col_in, sw_in, min_in, max_in, num_of_topics, ntop_w):
  # Vecotrize tokens
  if sw_in == "tf-idf":
    cv = TfidfVectorizer(stop_words=stoplist, ngram_range=(min_in, max_in))
  else:
    cv = CountVectorizer(stop_words=stoplist, ngram_range=(min_in, max_in))

  # Train model
  lda_model = LDA(n_components=num_of_topics)
  pipe = make_pipeline(cv, lda_model)
  pipe.fit(df_col_in)

  print_top_words(lda_model, cv.get_feature_names(), n_top_words=ntop_w)

  return lda_model

data = pd.read_csv("culture.csv")
stoplist = stopwords.words('english')
model = lda_model(data.comment, sw_in="tf-idf", min_in=2, max_in=3, num_of_topics=2, ntop_w=5)
