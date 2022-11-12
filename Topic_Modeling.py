# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation as LDA
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# reference
# https://towardsdatascience.com/nlp-with-lda-latent-dirichlet-allocation-and-text-clustering-to-improve-classification-97688c23d98
# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
# https://www.kaggle.com/code/ykhorramz/lda-and-t-sne-interactive-visualization/notebook


data = None # Update

# Add bigrams and trigrams (only ones that appear 10 times or more)
# topics are very similar what would make distinguish them are phrases rather than single/individual words.
def bigram_trigram(clean_df, min_count):
  docs = np.array(clean_df.comment)
  bigram = Phrases(docs, min_count=min_count)
  trigram = Phrases(bigram[docs])

  for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            docs[idx].append(token)
    for token in trigram[docs[idx]]:
        if '_' in token:
            docs[idx].append(token)
  return docs

def lda_model(docs, num_of_topics=1):
  # Create a dictionary representation of the documents.
  dictionary = Dictionary(docs)
  # Filter out words that occur less than 10 documents, or more than 20% of the documents.
  dictionary.filter_extremes(no_below=10, no_above=0.2)
  # vectorize doc
  corpus = [dictionary.doc2bow(doc) for doc in docs]
  # temp = dictionary[0]  # This is only to "load" the dictionary.
  # id2word = dictionary.id2token

  # build model
  lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_of_topics, \
                       random_state=100, eval_every=1)
  return lda_model

def explore_topic(lda_model, topic_number, topn, output=True):
    """
    accept a ldamodel, atopic number and topn vocabs of interest
    prints a formatted list of the topn terms
    """
    terms = []
    for term, frequency in lda_model.show_topic(topic_number, topn=topn):
        terms += [term]
        if output:
            print(u'{:20} {:.3f}'.format(term, round(frequency, 3)))

    return terms

topic_summaries = []
print(u'{:20} {}'.format(u'term', u'frequency') + u'\n')
for i in range(num_topics):
    print('Topic '+str(i)+'\n')
    tmp = explore_topic(model,topic_number=i, topn=10, output=True)
    topic_summaries += [tmp[:5]]
    print(tmp[:5])


# cross reference the topic generated with the topic column
# def lda_model(clean_df, num_of_topic=1):
#   count_vectorizer = CountVectorizer(stop_words='english')
#   transformed_clean_df = count_vectorizer.fit_transform(clean_df)
#   feature_names = count_vectorizer.get_feature_names()

#   lda = LDA(n_components=num_of_topic, learning_method='batch', random_state=2022)
#   # if the data size is large, the online update will be much faster than the batch update
#   lda.fit(transformed_clean_df)
#   return lda, feature_names

# def display_word_distribution(model, feature_names, n_word):
#     for topic_idx, topic in enumerate(model.components_):
#         print("Topic %d:" % (topic_idx))
#         words = []
#         for i in topic.argsort()[:-n_word - 1:-1]:
#             words.append(feature_names[i])
#         print(words)

# lda, features = lda_model(data)
# display_word_distribution(
#   model=lda_model, feature_names=features, n_word=10)