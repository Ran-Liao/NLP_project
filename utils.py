# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 20:31:11 2022

@author: pathouli
"""

def count_fun(var_in):
    tmp = var_in.split()
    return len(tmp)

def count_fun_unique(var_in):
    tmp = set(var_in.split())
    return len(tmp)

def clean_text(str_in):
    import re
    sent_a_clean = re.sub("[^A-Za-z']+", " ", str_in.lower()) 
    return sent_a_clean

def open_file(file_in):
    f = open(file_in, "r", encoding="utf-8")
    text = f.read()
    text = clean_text(text)
    f.close()
    return text

def file_reader(path_in):
    import os
    import pandas as pd
    the_data_t = pd.DataFrame()
    for root, dirs, files in os.walk(path_in, topdown=False):
       for name in files:
           try:
               label = root.split("/")[-1:][0]
               file_path = root + "/" + name
               text = open_file(file_path)
               if len(text) > 0:
                   the_data_t = the_data_t.append(
                       {"label": label, "body": text}, ignore_index=True)
           except:
               print (file_path)
               pass
    return the_data_t

def word_freq(df_in, col_in):
    import collections
    wrd_freq = dict()
    for topic in df_in.label.unique():
        tmp = df_in[df_in.label == topic]
        tmp_concat = tmp[col_in].str.cat(sep=" ")
        wrd_freq[topic] = collections.Counter(tmp_concat.split())
    return wrd_freq

def rem_sw(df_in):
    from nltk.corpus import stopwords
    sw = stopwords.words('english')
    sw.append("xp") #append a keyword to sw
    tmp = [word for word in df_in.split() if word not in sw]
    tmp = ' '.join(tmp)
    return tmp

def read_pickle(path_o, name_in):
    import pickle
    tmp_data = pickle.load(open(path_o + name_in + ".pk", "rb"))
    return tmp_data

def write_pickle(obj_in, path_o, name_in):
    import pickle
    pickle.dump(obj_in, open(path_o + name_in + ".pk", "wb"))
    return 0

def my_stem(var_in):
    #stemming
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    # example_sentence = "i was hiking down the trail towards by favorite fishing spot to catch lots of fishes"
    ex_stem = [ps.stem(word) for word in var_in.split()]
    ex_stem = ' '.join(ex_stem)
    return ex_stem

def sent_fun(str_in):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    senti = SentimentIntensityAnalyzer()
    ex = senti.polarity_scores(str_in)["compound"]
    return ex

def count_vec_fun(df_col_in, name_in, out_path_in, sw_in, min_in, max_in):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd
    if sw_in == "tf-idf":
        cv = TfidfVectorizer(ngram_range=(min_in, max_in))
    else:
        cv = CountVectorizer(ngram_range=(min_in, max_in))
    xform_data = pd.DataFrame(cv.fit_transform(df_col_in).toarray()) #be careful
    #takes up memory when force from sparse to dense
    xform_data.columns = cv.get_feature_names()
    write_pickle(cv, out_path_in, name_in)
    return xform_data

def chi_fun(df_in, label_in, name_in, out_path_in, num_feat):
    #chi-square
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import SelectKBest
    import pandas as pd
    feat_sel = SelectKBest(score_func=chi2, k=num_feat)
    dim_data = pd.DataFrame(feat_sel.fit_transform(df_in, label_in))
    feat_index = feat_sel.get_support(indices=True)
    feature_names = df_in.columns[feat_index]
    dim_data.columns = feature_names
    write_pickle(feat_sel, out_path_in, name_in)
    return dim_data

def cosine_fun(df_in, idx_in):
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    cos_sim = pd.DataFrame(cosine_similarity(df_in, df_in))
    cos_sim.index = idx_in
    cos_sim.columns = idx_in
    return cos_sim

def extract_embeddings_pre(df_in, num_vec_in, path_in, filename):
    #from gensim.models import Word2Vec
    import pandas as pd
    #from gensim.models import KeyedVectors
    import pickle
    #import gensim
    import gensim.downloader as api
    my_model = api.load(filename)
    
    #my_model = KeyedVectors.load_word2vec_format(
    #    filename, binary=True) 
    #my_model = Word2Vec(df_in.str.split(),
    #                    min_count=1, vector_size=300)
    #word_dict = my_model.wv.key_to_index
    #my_model.most_similar("calculus")
    #my_model.similarity("trout", "fish")
    def get_score(var):
        import numpy as np
        tmp_arr = list()
        try:
            for word in var:
                tmp_arr.append(list(my_model.get_vector(word)))
        except:
            tmp_arr.append(np.zeros(num_vec_in).tolist())
            #print(word)
            pass
        return np.mean(np.array(tmp_arr), axis=0)
    tmp_out = df_in.str.split().apply(get_score)
    tmp_data = tmp_out.apply(pd.Series).fillna(0)
    pickle.dump(my_model, open(path_in + "embeddings.pk", "wb"))
    pickle.dump(tmp_data, open(path_in + "embeddings_df.pk", "wb" ))
    return tmp_data

def extract_embeddings_domain(df_in, num_vec_in, path_in):
    #domain specific, train out own model specific to our domains
    from gensim.models import Word2Vec
    import pandas as pd
    import numpy as np
    import pickle
    model = Word2Vec(
        df_in.str.split(), min_count=1,
        vector_size=num_vec_in, workers=3, window=5, sg=0)
    wrd_dict = model.wv.key_to_index
    def get_score(var):
        try:
            tmp_arr = list()
            for word in var:
                tmp_arr.append(list(model.wv[word]))
        except:
            pass
        return np.mean(np.array(tmp_arr), axis=0)
    tmp_out = df_in.str.split().apply(get_score)
    tmp_data = tmp_out.apply(pd.Series).fillna(0)
    #model.wv.save_word2vec_format(path_in + "embeddings_domain.pk")
    pickle.dump(model, open(path_in + "embeddings_domain.pk", "wb"))
    pickle.dump(tmp_data, open(path_in + "embeddings_df_domain.pk", "wb" ))
    
    return tmp_data, wrd_dict

def pca_fun(df_in, exp_var_in, path_o, name_in):
    #pca
    from sklearn.decomposition import PCA
    import pandas as pd
    dim_red = PCA(n_components=exp_var_in)
    red_data = pd.DataFrame(dim_red.fit_transform(df_in))
    exp_var = sum(dim_red.explained_variance_ratio_)
    print ("Explained variance:", exp_var)
    write_pickle(dim_red, path_o, name_in)
    return red_data

def model_test_train_fun(df_in, label_in, test_size_in, path_in, xform_in):
    #TRAIN AN ALGO USING my_vec
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support
    import pandas as pd 
    my_model = RandomForestClassifier(random_state=123)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, label_in, test_size=test_size_in, random_state=42)
    
    #lets see how balanced the data is
    agg_cnts = pd.DataFrame(y_train).groupby('label')['label'].count()
    print (agg_cnts)
    
    my_model.fit(X_train, y_train)
    
    y_pred = my_model.predict(X_test)
    y_pred_proba = pd.DataFrame(my_model.predict_proba(X_test))
    y_pred_proba.columns = my_model.classes_
    
    metrics = pd.DataFrame(precision_recall_fscore_support(
        y_test, y_pred, average='weighted'))
    metrics.index = ["precision", "recall", "fscore", "none"]
    print (metrics)
    
    the_feats = read_pickle(path_in, xform_in)
    try:
        #feature importance
        fi = pd.DataFrame(my_model.feature_importances_)
        fi["feat_imp"] = the_feats.get_feature_names()
        fi.columns = ["feat_imp", "feature"]
        perc_propensity = len(fi[fi.feat_imp > 0]) / len(fi)
        print ("percent features that have propensity:", perc_propensity)
    except:
        print ("can't get features")
        pass
    return fi

def grid_fun(df_in, label_in, test_size_in, path_in, xform_in, grid_d, cv_in):
    #TRAIN AN ALGO USING my_vec
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support
    import pandas as pd 
    from sklearn.model_selection import GridSearchCV
    my_model = RandomForestClassifier(random_state=123)
    my_grid_model = GridSearchCV(my_model, param_grid=grid_d, cv=cv_in)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, label_in, test_size=test_size_in, random_state=42)
    
    my_grid_model.fit(X_train, y_train)
    
    print ("Best perf", my_grid_model.best_score_)
    print ("Best perf", my_grid_model.best_params_)
    
    my_model = RandomForestClassifier(
        **my_grid_model.best_params_, random_state=123)
    
    #lets see how balanced the data is
    agg_cnts = pd.DataFrame(y_train).groupby('label')['label'].count()
    print (agg_cnts)
    
    my_model.fit(X_train, y_train)
    write_pickle(my_model, path_in, "rf")
    
    y_pred = my_model.predict(X_test)
    y_pred_proba = pd.DataFrame(my_model.predict_proba(X_test))
    y_pred_proba.columns = my_model.classes_
    
    metrics = pd.DataFrame(precision_recall_fscore_support(
        y_test, y_pred, average='weighted'))
    metrics.index = ["precision", "recall", "fscore", "none"]
    print (metrics)
    
    the_feats = read_pickle(path_in, xform_in)
    try:
        #feature importance
        fi = pd.DataFrame(my_model.feature_importances_)
        fi["feat_imp"] = the_feats.get_feature_names()
        fi.columns = ["feat_imp", "feature"]
        perc_propensity = len(fi[fi.feat_imp > 0]) / len(fi)
        print ("percent features that have propensity:", perc_propensity)
    except:
        print ("can't get features")
        pass
    return fi

# def count_vec_fun(df_col_in, name_in, out_path_in):
#     from sklearn.feature_extraction.text import CountVectorizer
#     import pandas as pd
#     cv = CountVectorizer()
#     xform_data = pd.DataFrame(cv.fit_transform(df_col_in).toarray()) #be careful
#     col_names = cv.get_feature_names()
#     xform_data.columns = col_names
#     write_pickle(cv, out_path_in, name_in)
#     return xform_data