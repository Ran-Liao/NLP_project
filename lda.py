# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 18:59:46 2021

@author: pathouli
"""

import pickle
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import matplotlib.pyplot as plt
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import numpy as np
from funcs.utils import *
from kneed import KneeLocator

def fetch_bi_grams(var):
    sentence_stream = np.array(var)
    bigram = Phrases(sentence_stream, min_count=5, threshold=10, delimiter=",")
    trigram = Phrases(bigram[sentence_stream], min_count=5, threshold=10)
    bigram_phraser = Phraser(bigram)
    trigram_phraser = Phraser(trigram)
    bi_grams = list()
    tri_grams = list()
    for sent in sentence_stream:
        bi_grams.append(bigram_phraser[sent])
        tri_grams.append(trigram_phraser[sent])
    return bi_grams, tri_grams

def lda(df, col_in, max_ntopic=4, ntop_w=5):
    df["comment_clean"] = df[col_in].apply(clean_text).apply(rem_sw).str.split()

    bi, tri = fetch_bi_grams(df.comment_clean)
    the_data = bi
    dictionary = Dictionary(the_data)
    id2word = corpora.Dictionary(the_data)

    corpus = [id2word.doc2bow(text) for text in the_data]

    #compute Coherence Score using c_v
    coherence_model_lda = CoherenceModel(model=ldamodel, texts=the_data,
                                         dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    c_scores = list()
    for word in range(1, 8):
        ldamodel = gensim.models.ldamodel.LdaModel(
            corpus, num_topics=word, id2word=id2word, iterations=10, passes=5,
            random_state=123)
        coherence_model_lda = CoherenceModel(model=ldamodel, texts=the_data,
                                              dictionary=dictionary,
                                              coherence='c_v')
        c_scores.append(coherence_model_lda.get_coherence())

    x = range(1, max_ntopic)
    kn = KneeLocator(x, c_scores,
                     curve='concave', direction='increasing')
    opt_topics = kn.knee
    print ("Optimal topics is", opt_topics)

    # optimal model
    ldamodel = gensim.models.ldamodel.LdaModel(
        corpus, num_topics=opt_topics, id2word=id2word, iterations=50, passes=15,
        random_state=123)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=ntop_w)
    for topic in topics:
        print(topic)