#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:19:45 2022

@author: Dogar
"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt


################################
# Reading and appending the data
################################

the_path = "/Users/Dogar/Desktop/Columbia Material/Fall 2022/NLP for SS/Group Project/Reddit's Data Download/Data Download 2"


def read_csv_file(the_path):
    import os
    import pandas as pd
    the_data_t = pd.DataFrame()
    for root, dirs, files in os.walk(the_path, topdown=True):
          for name in files:
              try:
                  if not name.startswith('.'):
                      file_path = root + "/" + name
                      #print(filepath)
                      text = pd.read_csv(file_path)
                      the_data_t = the_data_t.append(text)
              except:
                  print (file_path)
                  pass
    return the_data_t

common_data = read_csv_file(the_path)

######################
# Filtering funciton
######################

def simple_search(str_a, str_b,):
   return common_data[(common_data['comment'].str.lower().str.contains(str_a)) & (common_data['comment'].str.lower().str.contains(str_b))]


######################
# Topic Modeling
######################

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

# Adding to the list of stopwords
stoplist = stopwords.words('english') + ['www', 'http', 'https', 'com', 'html', 'index'] + ['biden', 'politics']

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


######################
# Sentiment Analysis
######################


#This function takes in a string runs it through vader sentiment analysis and returns the compound score 
def vader_senti(str_in): 
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    senti = SentimentIntensityAnalyzer()
    out_put = senti.polarity_scores(str_in)["compound"]
    #out_put["sentiment"] = out_put.compound.apply(categorize_senti)
    return out_put


# Execution

def execute(str_a, str_b):
    
    # Filtering of the data
    data = simple_search(str_a, str_b)
    
    # Running the topic modelling model
    model = lda_model(data.comment, sw_in="tf-idf", min_in=2, max_in=3, num_of_topics=1, ntop_w=5)
    
    data["compound_score"] = data.comment.apply(vader_senti)


    # Ensure the date column is the datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Create a new column that collect just the month and date to allow for temporal grouping
    data['date_month'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m')


    data_plot = pd.DataFrame(data.groupby('date_month').mean().reset_index())


    date = data_plot['date_month']
    sentiment = data_plot['compound_score']
    plt.plot(date, sentiment)
    plt.title('Scatter Plot of average sentiment over time')
    plt.xlabel("Date")  # add X-axis label
    plt.ylabel("Sentiment (Compound Score)")  # add Y-axis label
    plt.xticks(rotation = 45) #change the orientation of x-axis for easy visibility
    plt.show()
    
execute("us", "politics")



